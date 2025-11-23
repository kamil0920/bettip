"""
Football API client with rate limiting and error handling.

Provides thread-safe rate-limited access to football API.
"""
import os
import time
import json
import threading
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, date
from pathlib import Path
from dotenv import load_dotenv
import logging


load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = os.getenv("API_BASE_URL", "https://v3.football.api-sports.io")
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", 7500))
PER_MIN_LIMIT = int(os.getenv("PER_MIN_LIMIT", 300))
STATE_PATH = os.getenv("STATE_PATH", "state.json")
HEADERS = {"x-apisports-key": API_KEY}


class TokenBucket:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, rate_per_min: int, burst: Optional[int] = None):
        self.rate_per_sec = rate_per_min / 60.0
        self.capacity = burst if burst is not None else rate_per_min
        self.tokens = self.capacity
        self.lock = threading.Lock()
        self.last = time.monotonic()

    def consume(self, tokens: float = 1.0) -> None:
        """Block until tokens are available, then consume them."""
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last
                add = elapsed * self.rate_per_sec
                if add > 0:
                    self.tokens = min(self.capacity, self.tokens + add)
                    self.last = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                needed = tokens - self.tokens
                to_wait = needed / self.rate_per_sec
            time.sleep(max(0.01, to_wait))


class FootballAPIClient:
    """
    Client for interacting with Football API with rate limiting and error handling.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.api_key = API_KEY
        self.base_url = BASE_URL
        self.daily_limit = DAILY_LIMIT
        self.per_min_limit = PER_MIN_LIMIT
        self.state_path = Path(STATE_PATH)

        self.bucket = TokenBucket(
            rate_per_min=self.per_min_limit,
            burst=self.per_min_limit
        )

        self.state = self._load_state()

        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _load_state(self) -> Dict[str, Any]:
        """Load persistent API state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    state = json.load(f)
                    if state.get("date") != str(date.today()):
                        state = {"date": str(date.today()), "count": 0}
                    return state
            except Exception as e:
                self.logger.warning(f"Failed to load API state: {e}")

        return {"date": str(date.today()), "count": 0}

    def _save_state(self) -> None:
        """Save persistent API state to disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save API state: {e}")

    def _check_daily_limit(self) -> None:
        """Check if daily API limit has been reached."""
        if self.state["count"] >= self.daily_limit:
            raise APIError(f"Daily limit {self.daily_limit} reached. Stop requests for today.")

    def _record_request(self) -> None:
        """Record an API request in persistent state."""
        self.state["count"] += 1
        self._save_state()

    def _make_request(
            self,
            endpoint: str,
            params: Dict[str, Any],
            max_retries: int = 5,
            timeout: int = 30
    ) -> Dict[str, Any]:
        """Make a rate-limited API request with robust error handling."""
        self._check_daily_limit()

        attempt = 0
        url = f"{self.base_url}{endpoint}"

        while attempt <= max_retries:
            self.bucket.consume(1.0)

            try:
                response = self.session.get(url, params=params, timeout=timeout)
            except requests.RequestException as e:
                backoff = min(60, 2 ** attempt)
                self.logger.warning(f"Request failed, retrying in {backoff}s: {e}")
                time.sleep(backoff)
                attempt += 1
                continue

            self._record_request()

            if response.status_code == 200:
                return response.json()

            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except (ValueError, TypeError):
                        wait = min(60, 2 ** attempt)
                else:
                    wait = min(60, 2 ** attempt)

                self.logger.info(f"Server error {response.status_code}, waiting {wait}s before retry")
                time.sleep(wait)
                attempt += 1
                continue

            response.raise_for_status()

        raise APIError(f"Exceeded max retries ({max_retries}) for {endpoint} with params {params}")

    def get_fixtures(self, league_id: int, season: int) -> List[Dict[str, Any]]:
        """Retrieve fixtures for a specific league and season."""
        try:
            response = self._make_request('/fixtures', {
                'league': league_id,
                'season': season
            })
            return response.get('response', [])
        except Exception as e:
            raise APIError(f"Failed to get fixtures: {str(e)}")

    def get_player_statistics(self, fixture_id: int) -> List[Dict[str, Any]]:
        """Retrieve player statistics for a specific fixture."""
        try:
            response = self._make_request('/players', {
                'fixture': fixture_id
            })
            return response.get('response', [])
        except Exception as e:
            raise APIError(f"Failed to get player statistics: {str(e)}")


class FootballPredictionError(Exception):
    """Base exception class for football prediction system."""
    pass


class APIError(FootballPredictionError):
    """Raised when API-related errors occur."""
    pass
