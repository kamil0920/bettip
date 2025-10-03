#!/usr/bin/env python3
"""
Weekly automation script for cron jobs.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from football_data_collector import weekly_update, LEAGUES_CONFIG

# Setup logging for cron
log_file = Path(__file__).parent / 'weekly_update.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)


def run_weekly_updates():
    """Run weekly updates for all configured leagues."""
    logger = logging.getLogger(__name__)

    logger.info("🔄 Starting weekly updates...")

    # Default to Premier League, but could be extended
    leagues_to_update = ['premier_league']  # Add more as needed

    results = {}

    for league in leagues_to_update:
        logger.info(f"📊 Updating {league}...")
        try:
            success = weekly_update(league)
            results[league] = success

            if success:
                logger.info(f"✅ {league} updated successfully")
            else:
                logger.error(f"❌ {league} update failed")

        except Exception as e:
            logger.error(f"❌ Error updating {league}: {e}")
            results[league] = False

    # Summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)

    logger.info(f"📊 Weekly update summary: {successful}/{total} successful")

    return successful == total


if __name__ == "__main__":
    success = run_weekly_updates()
    sys.exit(0 if success else 1)
