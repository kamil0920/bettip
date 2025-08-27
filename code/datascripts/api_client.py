import os
import time
import json
import requests
from datetime import date, datetime
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = os.getenv("API_BASE_URL", "https://v3.football.api-sports.io")
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "100"))
MIN_INTERVAL = float(os.getenv("MIN_INTERVAL_SECONDS", "6.0"))
STATE_PATH = "state.json"
HEADERS = {"x-apisports-key": API_KEY}

def _load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {"date": str(date.today()), "count": 0}

def _save_state(s):
    with open(STATE_PATH, "w") as f:
        json.dump(s, f)

class ApiFootballClient:
    def __init__(self):
        self.base = BASE_URL
        self.headers = HEADERS
        self.state = _load_state()
        if self.state.get("date") != str(date.today()):
            self.state = {"date": str(date.today()), "count": 0}
            _save_state(self.state)
        self.last_request_ts = None

    def _check_day_limit(self):
        if self.state["count"] >= DAILY_LIMIT:
            raise RuntimeError(f"Daily limit {DAILY_LIMIT} reached. Stop requests for today.")

    def _sleep_for_rate_limit(self):
        if self.last_request_ts is None:
            self.last_request_ts = time.time()
            return
        elapsed = time.time() - self.last_request_ts
        if elapsed < MIN_INTERVAL:
            to_sleep = MIN_INTERVAL - elapsed
            time.sleep(to_sleep)
        self.last_request_ts = time.time()

    def get(self, path: str, params: Dict[str, Any] = None, max_retries=4):
        self._check_day_limit()
        attempt = 0
        while attempt <= max_retries:
            self._sleep_for_rate_limit()
            url = f"{self.base}{path}"
            resp = requests.get(url, headers=self.headers, params=params, timeout=30)
            self.state["count"] += 1
            _save_state(self.state)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 503):
                backoff = 2 ** attempt
                time.sleep(backoff)
                attempt += 1
                continue
            resp.raise_for_status()
        raise RuntimeError("Exceeded max retries for GET " + path)
