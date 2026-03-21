"""Centralized notification system for bettip.

Usage:
    from src.notifications import TelegramNotifier, format_match_day_digest
    from src.notifications import REAL_ODDS_MARKETS, is_real_odds_market
"""

from src.notifications.formatters import (
    format_match_day_digest,
    format_post_optimization,
    format_pre_kickoff_update,
    format_weekly_report,
)
from src.notifications.market_utils import (
    REAL_ODDS_MARKETS,
    classify_recommendations,
    is_real_odds_market,
)
from src.notifications.telegram import TelegramNotifier

__all__ = [
    "REAL_ODDS_MARKETS",
    "TelegramNotifier",
    "classify_recommendations",
    "format_match_day_digest",
    "format_post_optimization",
    "format_pre_kickoff_update",
    "format_weekly_report",
    "is_real_odds_market",
]
