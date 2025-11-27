"""
Data collection module for fetching raw football data from APIs.

This module provides:
- API client with rate limiting and error handling
- Data collector for fixtures, lineups, events, and player statistics
- Scheduler for automated periodic updates
"""

from src.data_collection.api_client import (
    FootballAPIClient,
    TokenBucket,
    APIError,
    FootballPredictionError
)
from src.data_collection.collector import (
    FootballDataCollector,
    LEAGUES_CONFIG,
    bulk_collect
)

# Note: scheduler is imported directly where needed to avoid circular imports

__all__ = [
    # API Client
    'FootballAPIClient',
    'TokenBucket',
    'APIError',
    'FootballPredictionError',
    # Collector
    'FootballDataCollector',
    'LEAGUES_CONFIG',
    'bulk_collect',
]