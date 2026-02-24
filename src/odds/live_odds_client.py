"""Live Odds Client — fetches real-time niche market odds from The Odds API.

Replaces baseline (fake) odds with real bookmaker odds at inference time.
In-memory cache with 15-min TTL, quota tracking, graceful None fallback.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests as http_requests
from dotenv import load_dotenv

from src.odds.theodds_unified_loader import MARKET_KEYS, SPORT_KEYS

load_dotenv()

logger = logging.getLogger(__name__)

THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Re-use market key mappings from theodds_unified_loader (single source of truth).
# Keys: "btts", "corners", "cards", "shots", "h2h", "totals"
# Values: The Odds API market key strings
_STAT_TO_API_MARKET: Dict[str, str] = dict(MARKET_KEYS)

# Add new market mappings for team totals and handicaps
_STAT_TO_API_MARKET["hgoals"] = "alternate_team_totals"
_STAT_TO_API_MARKET["agoals"] = "alternate_team_totals"
_STAT_TO_API_MARKET["cornershc"] = "alternate_spreads_corners"
_STAT_TO_API_MARKET["cardshc"] = "alternate_spreads_cards"

# Stats with confirmed API coverage
_SUPPORTED_STATS = frozenset(_STAT_TO_API_MARKET.keys())

# Default cache TTL in seconds (15 minutes)
_DEFAULT_CACHE_TTL_SECONDS = 900

# Default quota safety threshold (stop fetching below this)
_DEFAULT_QUOTA_SAFETY_THRESHOLD = 50


@dataclass
class QuotaStatus:
    """Tracks API quota usage."""

    requests_used: int = 0
    requests_remaining: Optional[int] = None
    session_requests: int = 0
    last_checked: Optional[float] = None

    @property
    def is_exhausted(self) -> bool:
        """True if quota is known to be exhausted."""
        return self.requests_remaining is not None and self.requests_remaining <= 0


@dataclass
class CacheEntry:
    """Single cache entry with TTL."""

    data: Any
    timestamp: float

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if this entry has expired."""
        return (time.monotonic() - self.timestamp) > ttl_seconds


@dataclass
class MatchOdds:
    """Odds for a specific market line of a match."""

    over_avg: Optional[float] = None
    under_avg: Optional[float] = None
    over_max: Optional[float] = None
    under_max: Optional[float] = None
    line: Optional[float] = None
    available_lines: List[float] = field(default_factory=list)
    bookmaker_count: int = 0
    source: str = "the_odds_api"


def parse_market_name(market_name: str) -> Tuple[str, str, Optional[float]]:
    """Parse sniper market name into (stat, direction, line).

    Examples:
        'cards_under_35' -> ('cards', 'under', 3.5)
        'home_win'       -> ('h2h', 'home', None)
        'over25'         -> ('totals', 'over', 2.5)
    """
    if market_name == "home_win":
        return ("h2h", "home", None)
    if market_name == "away_win":
        return ("h2h", "away", None)
    if market_name in ("over25", "under25"):
        return ("totals", "over" if "over" in market_name else "under", 2.5)
    if market_name == "btts":
        return ("btts", "yes", None)
    if market_name == "double_chance_1x":
        return ("double_chance", "home_draw", None)
    if market_name == "double_chance_12":
        return ("double_chance", "home_away", None)
    if market_name == "double_chance_x2":
        return ("double_chance", "draw_away", None)

    parts = market_name.split("_")
    if len(parts) == 3:
        stat, direction, line_str = parts
        return (stat, direction, int(line_str) / 10.0)
    if len(parts) == 1 and parts[0] in _STAT_TO_API_MARKET:
        return (parts[0], "over", None)

    logger.warning("[LIVE ODDS] Cannot parse market name: %s", market_name)
    return (market_name, "over", None)


class LiveOddsClient:
    """Real-time odds client wrapping The Odds API v4 with TTL cache and quota tracking."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_seconds: float = _DEFAULT_CACHE_TTL_SECONDS,
        quota_safety_threshold: int = _DEFAULT_QUOTA_SAFETY_THRESHOLD,
        regions: str = "uk,eu",
        request_timeout: int = 30,
    ):
        """Initialize client.

        Args:
            api_key: The Odds API key. Falls back to THE_ODDS_API_KEY env var.
            cache_ttl_seconds: Cache TTL in seconds (default 900 = 15 minutes).
            quota_safety_threshold: Stop fetching below this many remaining requests.
            regions: Bookmaker regions to query (default 'uk,eu').
            request_timeout: HTTP request timeout in seconds.
        """
        self.api_key = api_key or os.getenv("THE_ODDS_API_KEY", "")
        self.cache_ttl_seconds = cache_ttl_seconds
        self.quota_safety_threshold = quota_safety_threshold
        self.regions = regions
        self.request_timeout = request_timeout

        self._cache: Dict[str, CacheEntry] = {}
        self._events_cache: Dict[str, CacheEntry] = {}
        self.quota = QuotaStatus()

        if not self.api_key:
            logger.warning(
                "[LIVE ODDS] No API key configured. "
                "Set THE_ODDS_API_KEY in .env or pass api_key parameter."
            )

    def get_match_odds(
        self,
        league: str,
        home_team: str,
        away_team: str,
        market: str,
    ) -> Optional[MatchOdds]:
        """Get real-time odds for a match and market. Returns None if unavailable."""
        if not self.api_key:
            logger.debug("[LIVE ODDS] No API key, returning None for %s", market)
            return None

        if not self._check_quota():
            return None

        sport_key = SPORT_KEYS.get(league)
        if not sport_key:
            logger.warning("[LIVE ODDS] Unknown league: %s", league)
            return None

        stat, direction, target_line = parse_market_name(market)
        api_market = _STAT_TO_API_MARKET.get(stat)
        if not api_market:
            # DEBUG not WARNING: fouls has no The Odds API market (known limitation)
            logger.debug(
                "[LIVE ODDS] No API market mapping for stat=%s (market=%s)",
                stat,
                market,
            )
            return None

        # Find the event
        event = self._find_event(sport_key, home_team, away_team)
        if not event:
            logger.debug(
                "[LIVE ODDS] Event not found: %s vs %s (%s)",
                home_team,
                away_team,
                league,
            )
            return None

        event_id = event["id"]

        # Fetch odds for this event + market (cached)
        raw_odds = self._fetch_event_market_odds(sport_key, event_id, api_market)
        if raw_odds is None:
            return None

        # Parse into MatchOdds based on market type
        if stat == "btts":
            return self._parse_btts(raw_odds)
        if stat == "h2h":
            return self._parse_h2h(raw_odds, event, direction)
        if stat == "totals":
            return self._parse_totals(raw_odds, "totals", target_line or 2.5)
        if stat == "double_chance":
            return self._parse_double_chance(raw_odds, direction)
        if stat == "goals":
            return self._parse_totals(raw_odds, "alternate_totals", target_line or 2.5)

        if stat in ("hgoals", "agoals"):
            return self._parse_team_totals(
                raw_odds, api_market, stat, target_line
            )
        if stat in ("cornershc", "cardshc"):
            return self._parse_spreads(
                raw_odds, api_market, target_line
            )

        # Niche totals (cards, corners, shots)
        return self._parse_totals(raw_odds, api_market, target_line)

    def get_batch_odds(
        self,
        league: str,
        home_team: str,
        away_team: str,
        markets: List[str],
    ) -> Dict[str, Optional[MatchOdds]]:
        """Get odds for multiple markets for one match. Event lookup is cached."""
        results: Dict[str, Optional[MatchOdds]] = {}
        for market in markets:
            results[market] = self.get_match_odds(league, home_team, away_team, market)
        return results

    def get_quota_status(self) -> QuotaStatus:
        """Return current quota status."""
        return self.quota

    def clear_cache(self) -> int:
        """Clear all cached data. Returns number of entries cleared."""
        count = len(self._cache) + len(self._events_cache)
        self._cache.clear()
        self._events_cache.clear()
        logger.info("[LIVE ODDS] Cache cleared (%d entries)", count)
        return count

    def evict_expired(self) -> int:
        """Remove expired entries from cache. Returns number evicted."""
        now = time.monotonic()
        ttl = self.cache_ttl_seconds
        stale = [k for k, v in self._cache.items() if (now - v.timestamp) > ttl]
        for k in stale:
            del self._cache[k]
        stale_ev = [
            k for k, v in self._events_cache.items() if (now - v.timestamp) > ttl
        ]
        for k in stale_ev:
            del self._events_cache[k]
        total = len(stale) + len(stale_ev)
        if total > 0:
            logger.debug("[LIVE ODDS] Evicted %d expired cache entries", total)
        return total

    @classmethod
    def get_supported_markets(cls) -> List[str]:
        """Return list of stat names that have confirmed API coverage."""
        return sorted(_SUPPORTED_STATS)

    def _check_quota(self) -> bool:
        """Check if we have enough quota to make requests."""
        if self.quota.is_exhausted:
            logger.warning("[LIVE ODDS] API quota exhausted, returning None")
            return False

        if (
            self.quota.requests_remaining is not None
            and self.quota.requests_remaining < self.quota_safety_threshold
        ):
            logger.warning(
                "[LIVE ODDS] Below safety threshold: %d remaining < %d threshold",
                self.quota.requests_remaining,
                self.quota_safety_threshold,
            )
            return False

        return True

    def _make_request(
        self, endpoint: str, params: Dict[str, str]
    ) -> Optional[Any]:
        """Make an API request with quota tracking. Returns JSON or None on failure."""
        params["apiKey"] = self.api_key
        url = f"{THE_ODDS_API_BASE}{endpoint}"

        try:
            response = http_requests.get(
                url, params=params, timeout=self.request_timeout
            )
            response.raise_for_status()
        except http_requests.exceptions.Timeout:
            logger.warning("[LIVE ODDS] Request timeout: %s", endpoint)
            return None
        except http_requests.exceptions.ConnectionError:
            logger.warning("[LIVE ODDS] Connection error: %s", endpoint)
            return None
        except http_requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            logger.warning(
                "[LIVE ODDS] HTTP %d for %s: %s", status, endpoint, exc
            )
            return None

        # Update quota from response headers
        self.quota.session_requests += 1
        remaining = response.headers.get("x-requests-remaining")
        used = response.headers.get("x-requests-used")
        if remaining is not None:
            self.quota.requests_remaining = int(remaining)
        if used is not None:
            self.quota.requests_used = int(used)
        self.quota.last_checked = time.monotonic()

        logger.debug(
            "[LIVE ODDS] API call: %s (used=%s, remaining=%s, session=%d)",
            endpoint,
            used,
            remaining,
            self.quota.session_requests,
        )

        try:
            return response.json()
        except (ValueError, TypeError, json.JSONDecodeError):
            logger.warning("[LIVE ODDS] Invalid JSON response for %s", endpoint)
            return None

    def _find_event(
        self, sport_key: str, home_team: str, away_team: str
    ) -> Optional[Dict[str, Any]]:
        """Find an event by team names. Tries exact match first, then token overlap."""
        events = self._get_events(sport_key)
        if not events:
            return None

        home_lower = home_team.lower().strip()
        away_lower = away_team.lower().strip()

        # Pass 1: exact match
        for event in events:
            ev_home = (event.get("home_team") or "").lower()
            ev_away = (event.get("away_team") or "").lower()
            if ev_home == home_lower and ev_away == away_lower:
                return event

        # Pass 2: token-overlap match (handles "Arsenal" vs "Arsenal FC",
        # "AC Milan" vs "Milan", etc.). Requires >= 50% token overlap on both sides.
        home_tokens = set(home_lower.split())
        away_tokens = set(away_lower.split())
        if home_tokens and away_tokens:
            best_event = None
            best_score = 0.0
            for event in events:
                ev_home = (event.get("home_team") or "").lower()
                ev_away = (event.get("away_team") or "").lower()
                ev_home_tokens = set(ev_home.split())
                ev_away_tokens = set(ev_away.split())
                if not ev_home_tokens or not ev_away_tokens:
                    continue
                home_overlap = len(home_tokens & ev_home_tokens) / max(
                    len(home_tokens), len(ev_home_tokens)
                )
                away_overlap = len(away_tokens & ev_away_tokens) / max(
                    len(away_tokens), len(ev_away_tokens)
                )
                if home_overlap >= 0.5 and away_overlap >= 0.5:
                    score = home_overlap + away_overlap
                    if score > best_score:
                        best_score = score
                        best_event = event
            if best_event is not None:
                logger.info(
                    "[LIVE ODDS] Token match: '%s' vs '%s' -> '%s' vs '%s'",
                    home_team,
                    away_team,
                    best_event.get("home_team"),
                    best_event.get("away_team"),
                )
                return best_event

        return None

    def _get_events(self, sport_key: str) -> List[Dict[str, Any]]:
        """Get upcoming events for a sport (cached)."""
        cache_key = f"events:{sport_key}"
        cached = self._events_cache.get(cache_key)
        if cached and not cached.is_expired(self.cache_ttl_seconds):
            return cached.data

        data = self._make_request(f"/sports/{sport_key}/events", {})
        if data is None:
            return []

        events = data if isinstance(data, list) else []
        self._events_cache[cache_key] = CacheEntry(
            data=events, timestamp=time.monotonic()
        )
        logger.debug(
            "[LIVE ODDS] Cached %d events for %s", len(events), sport_key
        )
        return events

    def _fetch_event_market_odds(
        self, sport_key: str, event_id: str, api_market: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch odds for a single event and market (cached)."""
        cache_key = f"odds:{event_id}:{api_market}"
        cached = self._cache.get(cache_key)
        if cached and not cached.is_expired(self.cache_ttl_seconds):
            return cached.data

        if not self._check_quota():
            return None

        params = {
            "regions": self.regions,
            "markets": api_market,
            "oddsFormat": "decimal",
        }
        data = self._make_request(
            f"/sports/{sport_key}/events/{event_id}/odds", params
        )
        if data is None:
            return None

        self._cache[cache_key] = CacheEntry(data=data, timestamp=time.monotonic())
        return data

    def _parse_totals(
        self,
        raw: Dict[str, Any],
        market_key: str,
        target_line: Optional[float],
    ) -> Optional[MatchOdds]:
        """Parse totals-style odds (corners, cards, shots, goals)."""
        all_lines: Dict[float, Dict[str, List[float]]] = {}
        for bookmaker in raw.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != market_key:
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price")
                    point = outcome.get("point")
                    if point is None or price is None:
                        continue
                    point = float(point)
                    if point not in all_lines:
                        all_lines[point] = {"over": [], "under": []}
                    if "Over" in name:
                        all_lines[point]["over"].append(float(price))
                    elif "Under" in name:
                        all_lines[point]["under"].append(float(price))

        if not all_lines:
            return None

        available = sorted(all_lines.keys())

        # Guard: reject if target line is far outside available range.
        # Prevents player_shots_on_target (lines ~4.5) from being mapped
        # to total match shots markets (lines ~24.5).
        if target_line is not None and available:
            max_avail = max(available)
            min_avail = min(available)
            if target_line > max_avail * 3 or target_line < min_avail / 3:
                logger.debug(
                    "[LIVE ODDS] Line mismatch: target=%.1f, available=[%.1f-%.1f]",
                    target_line,
                    min_avail,
                    max_avail,
                )
                return None

        # Find best matching line
        if target_line is not None:
            closest = min(available, key=lambda x: abs(x - target_line))
            # Reject if the exact requested line is not available.
            # Adjacent line odds (e.g., Over 2.5 @ 2.01 vs Over 1.5 @ 1.06)
            # are completely different and create phantom edges.
            if abs(closest - target_line) > 0.01:
                logger.info(
                    "[LIVE ODDS] Exact line %.1f not available, "
                    "closest=%.1f (available: %s) — rejecting",
                    target_line,
                    closest,
                    available,
                )
                return None
        else:
            closest = available[len(available) // 2]  # median line

        line_odds = all_lines[closest]
        over_prices = line_odds["over"]
        under_prices = line_odds["under"]

        if not over_prices and not under_prices:
            return None

        n_bookmakers = max(len(over_prices), len(under_prices))

        return MatchOdds(
            over_avg=_safe_mean(over_prices),
            under_avg=_safe_mean(under_prices),
            over_max=max(over_prices) if over_prices else None,
            under_max=max(under_prices) if under_prices else None,
            line=closest,
            available_lines=available,
            bookmaker_count=n_bookmakers,
        )

    def _parse_btts(self, raw: Dict[str, Any]) -> Optional[MatchOdds]:
        """Parse BTTS odds."""
        yes_odds: List[float] = []
        no_odds: List[float] = []

        for bookmaker in raw.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "btts":
                    continue
                for outcome in market.get("outcomes", []):
                    price = outcome.get("price")
                    if price is None:
                        continue
                    if outcome.get("name") == "Yes":
                        yes_odds.append(float(price))
                    elif outcome.get("name") == "No":
                        no_odds.append(float(price))

        if not yes_odds and not no_odds:
            return None

        return MatchOdds(
            over_avg=_safe_mean(yes_odds),
            under_avg=_safe_mean(no_odds),
            over_max=max(yes_odds) if yes_odds else None,
            under_max=max(no_odds) if no_odds else None,
            bookmaker_count=max(len(yes_odds), len(no_odds)),
        )

    def _parse_h2h(
        self,
        raw: Dict[str, Any],
        event: Dict[str, Any],
        direction: str,
    ) -> Optional[MatchOdds]:
        """Parse H2H (1X2) odds."""
        home_odds: List[float] = []
        draw_odds: List[float] = []
        away_odds: List[float] = []

        ev_home = event.get("home_team", "")
        ev_away = event.get("away_team", "")

        for bookmaker in raw.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price")
                    if price is None:
                        continue
                    if name == "Draw":
                        draw_odds.append(float(price))
                    elif name == ev_home:
                        home_odds.append(float(price))
                    elif name == ev_away:
                        away_odds.append(float(price))

        if not home_odds and not away_odds:
            return None

        primary = home_odds if direction == "home" else away_odds
        return MatchOdds(
            over_avg=_safe_mean(primary),
            under_avg=_safe_mean(draw_odds),
            bookmaker_count=len(primary),
        )


    def _parse_double_chance(
        self, raw: Dict[str, Any], direction: str
    ) -> Optional[MatchOdds]:
        """Parse Double Chance odds (Home/Draw, Home/Away, Draw/Away)."""
        home_draw_odds: List[float] = []
        home_away_odds: List[float] = []
        draw_away_odds: List[float] = []

        for bookmaker in raw.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "double_chance":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price")
                    if price is None:
                        continue
                    name_lower = name.lower().replace(" ", "")
                    if "home" in name_lower and "draw" in name_lower:
                        home_draw_odds.append(float(price))
                    elif "home" in name_lower and "away" in name_lower:
                        home_away_odds.append(float(price))
                    elif "draw" in name_lower and "away" in name_lower:
                        draw_away_odds.append(float(price))

        if not home_draw_odds and not home_away_odds and not draw_away_odds:
            return None

        # Select primary odds based on direction
        odds_map = {
            "home_draw": home_draw_odds,
            "home_away": home_away_odds,
            "draw_away": draw_away_odds,
        }
        primary = odds_map.get(direction, home_draw_odds)

        return MatchOdds(
            over_avg=_safe_mean(primary),
            bookmaker_count=len(primary),
        )


    def _parse_team_totals(
        self,
        raw: Dict[str, Any],
        market_key: str,
        stat: str,
        target_line: Optional[float],
    ) -> Optional[MatchOdds]:
        """Parse per-team goal totals odds."""
        home_team = raw.get("home_team", "")
        away_team = raw.get("away_team", "")
        is_home = stat == "hgoals"
        target_team = home_team if is_home else away_team

        all_lines: Dict[float, Dict[str, List[float]]] = {}
        for bookmaker in raw.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != market_key:
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    desc = (outcome.get("description") or "").strip()
                    price = outcome.get("price")
                    point = outcome.get("point")
                    if price is None or point is None or name != target_team:
                        continue
                    point = float(point)
                    if point not in all_lines:
                        all_lines[point] = {"over": [], "under": []}
                    if "Over" in desc:
                        all_lines[point]["over"].append(float(price))
                    elif "Under" in desc:
                        all_lines[point]["under"].append(float(price))

        if not all_lines:
            return None

        available = sorted(all_lines.keys())
        if target_line is not None:
            closest = min(available, key=lambda x: abs(x - target_line))
            if abs(closest - target_line) > 0.01:
                logger.info(
                    "[LIVE ODDS] Exact team totals line %.1f not available, "
                    "closest=%.1f — rejecting",
                    target_line,
                    closest,
                )
                return None
        else:
            closest = available[len(available) // 2]

        line_odds = all_lines[closest]
        over_prices = line_odds["over"]
        under_prices = line_odds["under"]
        if not over_prices and not under_prices:
            return None

        return MatchOdds(
            over_avg=_safe_mean(over_prices),
            under_avg=_safe_mean(under_prices),
            over_max=max(over_prices) if over_prices else None,
            under_max=max(under_prices) if under_prices else None,
            line=closest,
            available_lines=available,
            bookmaker_count=max(len(over_prices), len(under_prices)),
        )

    def _parse_spreads(
        self,
        raw: Dict[str, Any],
        market_key: str,
        target_line: Optional[float],
    ) -> Optional[MatchOdds]:
        """Parse handicap/spread odds, normalized to home team perspective."""
        home_team = raw.get("home_team", "")
        away_team = raw.get("away_team", "")

        all_lines: Dict[float, Dict[str, List[float]]] = {}
        for bookmaker in raw.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != market_key:
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price")
                    point = outcome.get("point")
                    if price is None or point is None:
                        continue
                    point = float(point)
                    abs_line = abs(point)
                    if abs_line not in all_lines:
                        all_lines[abs_line] = {"over": [], "under": []}

                    if name == home_team:
                        if point < 0:
                            all_lines[abs_line]["over"].append(float(price))
                        else:
                            all_lines[abs_line]["under"].append(float(price))
                    elif name == away_team:
                        if point > 0:
                            all_lines[abs_line]["under"].append(float(price))
                        else:
                            all_lines[abs_line]["over"].append(float(price))

        if not all_lines:
            return None

        available = sorted(all_lines.keys())
        if target_line is not None:
            closest = min(available, key=lambda x: abs(x - target_line))
            if abs(closest - target_line) > 0.01:
                logger.info(
                    "[LIVE ODDS] Exact spread line %.1f not available, "
                    "closest=%.1f — rejecting",
                    target_line,
                    closest,
                )
                return None
        else:
            closest = available[len(available) // 2]

        line_odds = all_lines[closest]
        over_prices = line_odds["over"]
        under_prices = line_odds["under"]
        if not over_prices and not under_prices:
            return None

        return MatchOdds(
            over_avg=_safe_mean(over_prices),
            under_avg=_safe_mean(under_prices),
            over_max=max(over_prices) if over_prices else None,
            under_max=max(under_prices) if under_prices else None,
            line=closest,
            available_lines=available,
            bookmaker_count=max(len(over_prices), len(under_prices)),
        )


def _safe_mean(values: List[float]) -> Optional[float]:
    """Compute mean of a list, returning None for empty lists."""
    if not values:
        return None
    return sum(values) / len(values)


# Pipeline column name mappings for generate_daily_recommendations.py integration
_PIPELINE_ODDS_MAP: Dict[str, Tuple[str, str]] = {
    "home_win": ("h2h_home_avg", "h2h_draw_avg"),
    "away_win": ("h2h_away_avg", "h2h_draw_avg"),
    "over25": ("totals_over_avg", "totals_under_avg"),
    "under25": ("totals_under_avg", "totals_over_avg"),
    "btts": ("btts_yes_avg", "btts_no_avg"),
}

# Double chance uses a single odds value (no 2-way complement)
_DC_PIPELINE_MAP: Dict[str, str] = {
    "double_chance_1x": "dc_home_draw_avg",
    "double_chance_12": "dc_home_away_avg",
    "double_chance_x2": "dc_draw_away_avg",
}


def to_pipeline_odds(market_name: str, odds: MatchOdds) -> Dict[str, float]:
    """Map MatchOdds to pipeline column names for calculate_edge().

    For H2H/totals/btts markets, maps to existing column names.
    For niche markets, maps to '{stat}_over_avg' / '{stat}_under_avg'.
    """
    result: Dict[str, float] = {}
    if market_name in _PIPELINE_ODDS_MAP:
        over_col, under_col = _PIPELINE_ODDS_MAP[market_name]
        if odds.over_avg is not None:
            result[over_col] = odds.over_avg
        if odds.under_avg is not None:
            result[under_col] = odds.under_avg
        return result

    # Double chance: single odds value, no complement
    if market_name in _DC_PIPELINE_MAP:
        dc_col = _DC_PIPELINE_MAP[market_name]
        if odds.over_avg is not None:
            result[dc_col] = odds.over_avg
        return result

    # Niche markets: column names follow pattern "{stat}_over_avg" / "{stat}_under_avg"
    stat, _, line = parse_market_name(market_name)
    # Generic columns (backward compat + base market usage)
    if odds.over_avg is not None:
        result[f"{stat}_over_avg"] = odds.over_avg
    if odds.under_avg is not None:
        result[f"{stat}_under_avg"] = odds.under_avg
    # Per-line columns for line-specific markets (e.g., corners_over_avg_85)
    if line is not None:
        line_key = str(line).replace(".", "")
        if odds.over_avg is not None:
            result[f"{stat}_over_avg_{line_key}"] = odds.over_avg
        if odds.under_avg is not None:
            result[f"{stat}_under_avg_{line_key}"] = odds.under_avg
    return result
