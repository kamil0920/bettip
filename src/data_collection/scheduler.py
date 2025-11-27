"""
Scheduler for automated data collection tasks.

Provides utilities for running periodic updates of football data.
"""
import sys
import logging
from datetime import datetime
from typing import List, Dict, Optional

from src.data_collection.collector import FootballDataCollector, LEAGUES_CONFIG

logger = logging.getLogger(__name__)


def weekly_update(
        league_key: str = 'premier_league',
        base_data_dir: str = "data/01-raw",
        include_lineups: bool = True,
        include_events: bool = True,
        include_player_stats: bool = True
) -> bool:
    """
    Perform weekly update for a specific league.

    Collects data for the current season including recent fixtures,
    lineups, events, and player statistics.

    Args:
        league_key: League identifier
        base_data_dir: Base directory for raw data
        include_lineups: Whether to collect lineups
        include_events: Whether to collect events
        include_player_stats: Whether to collect player statistics

    Returns:
        True if update was successful
    """
    collector = FootballDataCollector(base_data_dir)
    current_season = datetime.now().year

    logger.info(f"Running weekly update for {league_key} season {current_season}")

    try:
        # Update fixtures first
        success = collector.collect_season_data(league_key, current_season, force_refresh=True)

        if not success:
            logger.error(f"Failed to update fixtures for {league_key}")
            return False

        # Collect additional data
        if include_lineups:
            collector.collect_match_lineups(league_key, current_season, completed_only=True)

        if include_events:
            collector.collect_match_events(league_key, current_season, completed_only=True)

        if include_player_stats:
            collector.collect_player_statistics(league_key, current_season, completed_only=True)

        logger.info(f"Weekly update completed for {league_key}")
        return True

    except Exception as e:
        logger.error(f"Error during weekly update for {league_key}: {e}")
        return False


def run_scheduled_updates(
        leagues: Optional[List[str]] = None,
        base_data_dir: str = "data/01-raw"
) -> Dict[str, bool]:
    """
    Run scheduled updates for multiple leagues.

    Args:
        leagues: List of league keys to update (defaults to Premier League)
        base_data_dir: Base directory for raw data

    Returns:
        Dictionary mapping league keys to success status
    """
    if leagues is None:
        leagues = ['premier_league']

    logger.info("Starting scheduled updates...")

    results = {}

    for league in leagues:
        if league not in LEAGUES_CONFIG:
            logger.warning(f"Unknown league: {league}, skipping")
            results[league] = False
            continue

        logger.info(f"Updating {league}...")

        try:
            success = weekly_update(league, base_data_dir)
            results[league] = success

            if success:
                logger.info(f"{league} updated successfully")
            else:
                logger.error(f"{league} update failed")

        except Exception as e:
            logger.error(f"Error updating {league}: {e}")
            results[league] = False

    # Summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)

    logger.info(f"Scheduled update summary: {successful}/{total} successful")

    return results


def main() -> int:
    """
    Main entry point for scheduled updates.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    results = run_scheduled_updates()

    all_successful = all(results.values())
    return 0 if all_successful else 1


if __name__ == "__main__":
    sys.exit(main())
