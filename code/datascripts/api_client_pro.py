import os
import time
import json
import threading
import requests
from typing import Dict, Any, Optional
from datetime import date
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = os.getenv("API_BASE_URL", "https://v3.football.api-sports.io")
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "7500"))
PER_MIN_LIMIT = int(os.getenv("PER_MIN_LIMIT", "300"))
STATE_PATH = os.getenv("STATE_PATH", "state.json")
HEADERS = {"x-apisports-key": API_KEY}

# Token bucket rate limiter
class TokenBucket:
    def __init__(self, rate_per_min: int, burst: Optional[int] = None):
        self.rate_per_sec = rate_per_min / 60.0
        self.capacity = burst if burst is not None else rate_per_min
        self.tokens = self.capacity
        self.lock = threading.Lock()
        self.last = time.monotonic()

    def consume(self, tokens: float = 1.0):
        """
        Block until tokens are available, then consume them.
        Safe to call from multiple threads.
        """
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
                # compute approximate time to wait
                needed = tokens - self.tokens
                to_wait = needed / self.rate_per_sec
            # sleep outside lock
            time.sleep(max(0.01, to_wait))

# persistent simple state to track daily requests and some metadata
def _load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {"date": str(date.today()), "count": 0, "batches": {}}
    return {"date": str(date.today()), "count": 0, "batches": {}}

def _save_state(s):
    with open(STATE_PATH, "w") as f:
        json.dump(s, f, indent=2)

class ApiFootballClient:
    def __init__(self, base_url: str = None, per_min_limit: int = PER_MIN_LIMIT, daily_limit: int = DAILY_LIMIT):
        self.base = base_url or BASE_URL
        self.headers = HEADERS
        self.daily_limit = daily_limit
        self.state = _load_state()
        if self.state.get("date") != str(date.today()):
            self.state = {"date": str(date.today()), "count": 0, "batches": {}}
            _save_state(self.state)
        # token bucket rate limiter
        self.bucket = TokenBucket(rate_per_min=per_min_limit, burst=per_min_limit)
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _check_day_limit(self):
        if self.state["count"] >= self.daily_limit:
            raise RuntimeError(f"Daily limit {self.daily_limit} reached. Stop requests for today.")

    def _record_request(self):
        self.state["count"] += 1
        _save_state(self.state)

    def get(self, path: str, params: Dict[str, Any] = None, max_retries: int = 5, timeout: int = 30, return_response_headers: bool = False):
        """
        Thread-safe GET that respects token bucket and daily limit.
        Returns JSON on success. Optionally return (json, headers).
        """
        self._check_day_limit()
        attempt = 0
        while attempt <= max_retries:
            # block until token available
            self.bucket.consume(1.0)
            url = f"{self.base}{path}"
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
            except requests.RequestException as e:
                backoff = min(60, 2 ** attempt)
                attempt += 1
                time.sleep(backoff)
                continue

            # record request count (after making request)
            self._record_request()

            # success
            if resp.status_code == 200:
                if return_response_headers:
                    return resp.json(), resp.headers
                return resp.json()

            # rate limit or server error -> backoff and retry
            if resp.status_code in (429, 500, 502, 503, 504):
                # try to honor Retry-After header if present
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except Exception:
                        wait = 2 ** attempt
                else:
                    wait = min(60, 2 ** attempt)
                time.sleep(wait)
                attempt += 1
                continue

            # other client error - raise
            resp.raise_for_status()

        raise RuntimeError(f"Exceeded max retries for GET {path} with params {params}")
