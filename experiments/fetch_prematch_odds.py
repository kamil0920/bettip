#!/usr/bin/env python3
"""
Fetch pre-match odds from The Odds API for all leagues.

Fetches h2h (1X2) and totals (Over/Under 2.5) odds for upcoming fixtures,
plus niche markets (BTTS, corners, shots). Saves combined odds to parquet
for use by prediction scripts.

Usage:
    python experiments/fetch_prematch_odds.py
    python experiments/fetch_prematch_odds.py --leagues premier_league bundesliga
    python experiments/fetch_prematch_odds.py --hours 24          # Only next 24h (default)
    python experiments/fetch_prematch_odds.py --hours 0           # All upcoming (expensive!)
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.odds.theodds_unified_loader import (
    SPORT_KEYS,
    TheOddsUnifiedLoader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_LEAGUES = list(SPORT_KEYS.keys())

OUTPUT_DIR = project_root / "data" / "prematch_odds"


def fetch_prematch_odds(
    leagues: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    max_hours_ahead: int = 24,
) -> pd.DataFrame:
    """
    Fetch pre-match odds for all markets and leagues.

    Args:
        leagues: Leagues to fetch (default: all configured leagues).
        output_dir: Directory to save parquet output.
        max_hours_ahead: Only fetch events within this time window.
            Default 24h. Set to 0 for all upcoming (expensive!).

    Returns:
        Combined DataFrame with all odds across leagues.
    """
    if leagues is None:
        leagues = DEFAULT_LEAGUES
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    loader = TheOddsUnifiedLoader()

    status = loader.check_api_status()
    if status.get("status") != "ok":
        logger.error(f"API not available: {status}")
        return pd.DataFrame()

    logger.info(
        f"API status: {status.get('requests_used')} used, "
        f"{status.get('requests_remaining')} remaining"
    )

    hours_filter = max_hours_ahead if max_hours_ahead > 0 else None
    if hours_filter:
        logger.info(f"Time filter: only events within {hours_filter}h")
    else:
        logger.info("No time filter — fetching ALL upcoming events (expensive!)")

    all_odds: list[pd.DataFrame] = []

    for league in leagues:
        if league not in SPORT_KEYS:
            logger.warning(f"Unknown league: {league}, skipping")
            continue

        logger.info(f"Fetching odds for {league}...")
        try:
            df = loader.fetch_all_markets(
                league, max_hours_ahead=hours_filter
            )
            if not df.empty:
                logger.info(
                    f"  {league}: {len(df)} matches, "
                    f"h2h={df['h2h_home_avg'].notna().sum() if 'h2h_home_avg' in df.columns else 0}, "
                    f"totals={df['totals_over_avg'].notna().sum() if 'totals_over_avg' in df.columns else 0}, "
                    f"btts={df['btts_yes_avg'].notna().sum() if 'btts_yes_avg' in df.columns else 0}"
                )
                all_odds.append(df)
            else:
                logger.info(f"  {league}: no upcoming matches")
        except Exception as e:
            logger.error(f"  {league}: failed - {e}")

    if not all_odds:
        logger.warning("No odds fetched for any league")
        return pd.DataFrame()

    combined = pd.concat(all_odds, ignore_index=True)

    # Save dated parquet
    date_str = datetime.now().strftime("%Y%m%d")
    output_file = output_dir / f"odds_{date_str}.parquet"
    combined.to_parquet(output_file, index=False)
    logger.info(f"Saved {len(combined)} matches to {output_file}")

    # Also save CSV for debugging
    csv_file = output_dir / f"odds_{date_str}.csv"
    combined.to_csv(csv_file, index=False)

    # Save a "latest" symlink-like copy for easy loading
    latest_file = output_dir / "odds_latest.parquet"
    combined.to_parquet(latest_file, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"PREMATCH ODDS FETCH SUMMARY")
    print(f"{'='*60}")
    print(f"Total matches: {len(combined)}")
    print(f"Leagues: {combined['league'].unique().tolist()}")

    for market, col in [
        ("H2H (1X2)", "h2h_home_avg"),
        ("Totals (O/U)", "totals_over_avg"),
        ("BTTS", "btts_yes_avg"),
        ("Corners", "corners_over_avg"),
        ("Shots", "shots_over_avg"),
        ("Cards", "cards_over_avg"),
        ("Goals (alt)", "goals_over_avg"),
        ("Double Chance", "dc_home_draw_avg"),
    ]:
        if col in combined.columns:
            count = combined[col].notna().sum()
            print(f"  {market}: {count}/{len(combined)} matches")

    print(f"\nOutput: {output_file}")
    print(f"API requests remaining: {loader.requests_remaining}")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Fetch pre-match odds")
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=None,
        help="Leagues to fetch (default: all)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Only fetch events within N hours (default: 24, 0 = all upcoming)",
    )
    args = parser.parse_args()

    fetch_prematch_odds(leagues=args.leagues, max_hours_ahead=args.hours)


if __name__ == "__main__":
    main()
