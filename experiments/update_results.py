#!/usr/bin/env python3
"""
Update results for previous predictions - settle bets and record closing odds for CLV.

Called by prematch-intelligence GitHub Actions workflow before new predictions.

Usage:
    python experiments/update_results.py
"""
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.ml.clv_tracker import CLVTracker
from src.recommendations.generator import load_recommendations

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLV_OUTPUT_DIR = str(project_root / "data" / "04-predictions" / "clv_tracking")
RECOMMENDATIONS_DIR = project_root / "data" / "05-recommendations"


def load_match_results() -> pd.DataFrame:
    """Load match results from raw data to settle bets."""
    all_matches = []
    raw_dir = project_root / "data" / "01-raw"

    for league_dir in raw_dir.iterdir():
        if not league_dir.is_dir():
            continue
        for season_dir in league_dir.iterdir():
            matches_file = season_dir / "matches.parquet"
            if matches_file.exists():
                df = pd.read_parquet(matches_file)
                all_matches.append(df)

    if not all_matches:
        return pd.DataFrame()

    return pd.concat(all_matches, ignore_index=True)


def load_match_stats() -> pd.DataFrame:
    """Load match stats for niche market settlement."""
    all_stats = []
    raw_dir = project_root / "data" / "01-raw"

    for league_dir in raw_dir.iterdir():
        if not league_dir.is_dir():
            continue
        for season_dir in league_dir.iterdir():
            stats_file = season_dir / "match_stats.parquet"
            if stats_file.exists():
                df = pd.read_parquet(stats_file)
                all_stats.append(df)

    if not all_stats:
        return pd.DataFrame()

    return pd.concat(all_stats, ignore_index=True)


def settle_recommendation(row: pd.Series, matches: pd.DataFrame, stats: pd.DataFrame) -> dict | None:
    """Determine if a recommendation won or lost based on match results."""
    fixture_id = row.get("fixture_id")
    if pd.isna(fixture_id):
        return None

    fixture_id = int(fixture_id)
    market = str(row.get("market", "")).upper()
    side = str(row.get("side", "")).upper()
    line = row.get("line", 0)

    # Match result markets
    if market in ("HOME_WIN", "AWAY_WIN"):
        match = matches[matches["fixture.id"] == fixture_id]
        if match.empty or match.iloc[0].get("fixture.status.short") != "FT":
            return None

        home_goals = match.iloc[0]["goals.home"]
        away_goals = match.iloc[0]["goals.away"]

        if market == "HOME_WIN":
            won = home_goals > away_goals
        else:
            won = away_goals > home_goals

        return {"won": won, "actual_value": f"{home_goals}-{away_goals}"}

    # Goals markets
    if market in ("OVER_2.5", "UNDER_2.5"):
        match = matches[matches["fixture.id"] == fixture_id]
        if match.empty or match.iloc[0].get("fixture.status.short") != "FT":
            return None

        total_goals = match.iloc[0]["goals.home"] + match.iloc[0]["goals.away"]

        if market == "OVER_2.5":
            won = total_goals > 2.5
        else:
            won = total_goals < 2.5

        return {"won": won, "actual_value": total_goals}

    # BTTS
    if market == "BTTS":
        match = matches[matches["fixture.id"] == fixture_id]
        if match.empty or match.iloc[0].get("fixture.status.short") != "FT":
            return None

        btts = match.iloc[0]["goals.home"] > 0 and match.iloc[0]["goals.away"] > 0
        won = btts if side == "YES" else not btts
        return {"won": won, "actual_value": btts}

    # Niche markets (stats-based)
    stat_map = {
        "FOULS": ("home_fouls", "away_fouls"),
        "SHOTS": ("home_shots_total", "away_shots_total"),
        "CORNERS": ("home_corners", "away_corners"),
        "CARDS": ("home_yellow_cards", "away_yellow_cards"),
    }

    if market in stat_map and not stats.empty:
        match_stats = stats[stats["fixture_id"] == fixture_id]
        if match_stats.empty:
            return None

        home_col, away_col = stat_map[market]
        if home_col not in match_stats.columns or away_col not in match_stats.columns:
            return None

        total = match_stats.iloc[0][home_col] + match_stats.iloc[0][away_col]

        if side == "OVER":
            won = total > line
        else:
            won = total < line

        return {"won": won, "actual_value": total}

    return None


def main() -> None:
    tracker = CLVTracker(output_dir=CLV_OUTPUT_DIR)
    logger.info(f"CLV tracker: {len(tracker.predictions)} predictions loaded")

    # Find unsettled recommendation CSVs (last 7 days)
    rec_files = sorted(RECOMMENDATIONS_DIR.glob("rec_*.csv"))
    if not rec_files:
        logger.info("No recommendation files found")
        return

    # Load match data
    matches = load_match_results()
    stats = load_match_stats()

    if matches.empty:
        logger.info("No match results available")
        return

    logger.info(f"Loaded {len(matches)} match results, {len(stats)} stat records")

    settled_count = 0
    for rec_file in rec_files:
        try:
            df = pd.read_csv(rec_file)
        except Exception as e:
            logger.warning(f"Failed to read {rec_file.name}: {e}")
            continue

        if "status" not in df.columns:
            df["status"] = "PENDING"
        pending = df[df["status"] == "PENDING"]
        if pending.empty:
            continue

        updated = False
        for idx, row in pending.iterrows():
            result = settle_recommendation(row, matches, stats)
            if result is None:
                continue

            # Update CSV
            df.at[idx, "status"] = "WON" if result["won"] else "LOST"
            df.at[idx, "actual_value"] = result["actual_value"]
            df.at[idx, "settled_at"] = datetime.now().isoformat()

            odds = row.get("odds", 0)
            if result["won"] and odds > 0:
                df.at[idx, "pnl"] = odds - 1
            elif not result["won"]:
                df.at[idx, "pnl"] = -1

            # Update CLV tracker
            fixture_id = str(int(row["fixture_id"]))
            market = str(row.get("market", "")).lower().replace(".", "").replace("_", "")
            market_map = {
                "over25": "over25", "under25": "under25",
                "homewin": "home_win", "awaywin": "away_win",
                "btts": "btts", "corners": "corners",
                "fouls": "fouls", "shots": "shots", "cards": "cards",
            }
            bet_type = market_map.get(market, market)
            tracker.record_result(match_id=fixture_id, bet_type=bet_type, won=result["won"])

            settled_count += 1
            updated = True

        if updated:
            df.to_csv(rec_file, index=False)
            logger.info(f"Updated {rec_file.name}")

    tracker.save_history()
    logger.info(f"Settled {settled_count} bets total")


if __name__ == "__main__":
    main()
