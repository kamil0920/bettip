"""Unit tests for the live odds client module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.odds.live_odds_client import (
    CacheEntry,
    LiveOddsClient,
    MatchOdds,
    QuotaStatus,
    _safe_mean,
    parse_market_name,
    to_pipeline_odds,
)


# --------------------------------------------------------------------------- #
# parse_market_name tests
# --------------------------------------------------------------------------- #


class TestParseMarketName:
    """Tests for the market name parser."""

    def test_cards_under(self):
        stat, direction, line = parse_market_name("cards_under_35")
        assert stat == "cards"
        assert direction == "under"
        assert line == 3.5

    def test_corners_over(self):
        stat, direction, line = parse_market_name("corners_over_95")
        assert stat == "corners"
        assert direction == "over"
        assert line == 9.5

    def test_shots_over_high_line(self):
        stat, direction, line = parse_market_name("shots_over_265")
        assert stat == "shots"
        assert direction == "over"
        assert line == 26.5

    def test_fouls_under(self):
        stat, direction, line = parse_market_name("fouls_under_255")
        assert stat == "fouls"
        assert direction == "under"
        assert line == 25.5

    def test_home_win(self):
        stat, direction, line = parse_market_name("home_win")
        assert stat == "h2h"
        assert direction == "home"
        assert line is None

    def test_away_win(self):
        stat, direction, line = parse_market_name("away_win")
        assert stat == "h2h"
        assert direction == "away"
        assert line is None

    def test_over25(self):
        stat, direction, line = parse_market_name("over25")
        assert stat == "totals"
        assert direction == "over"
        assert line == 2.5

    def test_under25(self):
        stat, direction, line = parse_market_name("under25")
        assert stat == "totals"
        assert direction == "under"
        assert line == 2.5

    def test_btts(self):
        stat, direction, line = parse_market_name("btts")
        assert stat == "btts"
        assert direction == "yes"
        assert line is None

    def test_base_niche_market(self):
        stat, direction, line = parse_market_name("shots")
        assert stat == "shots"
        assert direction == "over"
        assert line is None

    def test_cards_over_15(self):
        stat, direction, line = parse_market_name("cards_over_15")
        assert stat == "cards"
        assert direction == "over"
        assert line == 1.5

    def test_corners_under_115(self):
        stat, direction, line = parse_market_name("corners_under_115")
        assert stat == "corners"
        assert direction == "under"
        assert line == 11.5


# --------------------------------------------------------------------------- #
# CacheEntry tests
# --------------------------------------------------------------------------- #


class TestCacheEntry:
    """Tests for cache entry TTL logic."""

    def test_not_expired(self):
        entry = CacheEntry(data="test", timestamp=time.monotonic())
        assert not entry.is_expired(ttl_seconds=900)

    def test_expired(self):
        entry = CacheEntry(data="test", timestamp=time.monotonic() - 1000)
        assert entry.is_expired(ttl_seconds=900)

    def test_zero_ttl_always_expired(self):
        entry = CacheEntry(data="test", timestamp=time.monotonic())
        assert entry.is_expired(ttl_seconds=0)

    def test_just_within_ttl(self):
        entry = CacheEntry(data="test", timestamp=time.monotonic() - 899)
        assert not entry.is_expired(ttl_seconds=900)


# --------------------------------------------------------------------------- #
# QuotaStatus tests
# --------------------------------------------------------------------------- #


class TestQuotaStatus:
    """Tests for quota tracking."""

    def test_initial_not_exhausted(self):
        qs = QuotaStatus()
        assert not qs.is_exhausted

    def test_exhausted_at_zero(self):
        qs = QuotaStatus(requests_remaining=0)
        assert qs.is_exhausted

    def test_not_exhausted_with_remaining(self):
        qs = QuotaStatus(requests_remaining=100)
        assert not qs.is_exhausted

    def test_exhausted_negative(self):
        qs = QuotaStatus(requests_remaining=-1)
        assert qs.is_exhausted


# --------------------------------------------------------------------------- #
# _safe_mean tests
# --------------------------------------------------------------------------- #


class TestSafeMean:
    """Tests for the safe mean helper."""

    def test_empty_returns_none(self):
        assert _safe_mean([]) is None

    def test_single_value(self):
        assert _safe_mean([3.5]) == 3.5

    def test_multiple_values(self):
        assert _safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_decimal_precision(self):
        result = _safe_mean([1.85, 1.90, 1.95])
        assert result == pytest.approx(1.9, abs=0.01)


# --------------------------------------------------------------------------- #
# LiveOddsClient tests
# --------------------------------------------------------------------------- #

# Fake API responses for testing
_FAKE_EVENTS_RESPONSE = [
    {
        "id": "evt_001",
        "sport_key": "soccer_epl",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "commence_time": "2026-02-20T15:00:00Z",
    },
    {
        "id": "evt_002",
        "sport_key": "soccer_epl",
        "home_team": "Liverpool",
        "away_team": "Manchester United",
        "commence_time": "2026-02-20T17:30:00Z",
    },
]

_FAKE_TOTALS_RESPONSE = {
    "id": "evt_001",
    "bookmakers": [
        {
            "key": "bet365",
            "title": "Bet365",
            "markets": [
                {
                    "key": "alternate_totals_cards",
                    "outcomes": [
                        {"name": "Over", "price": 1.85, "point": 3.5},
                        {"name": "Under", "price": 1.95, "point": 3.5},
                        {"name": "Over", "price": 2.20, "point": 4.5},
                        {"name": "Under", "price": 1.65, "point": 4.5},
                    ],
                }
            ],
        },
        {
            "key": "williamhill",
            "title": "William Hill",
            "markets": [
                {
                    "key": "alternate_totals_cards",
                    "outcomes": [
                        {"name": "Over", "price": 1.80, "point": 3.5},
                        {"name": "Under", "price": 2.00, "point": 3.5},
                    ],
                }
            ],
        },
    ],
}

_FAKE_BTTS_RESPONSE = {
    "id": "evt_001",
    "bookmakers": [
        {
            "key": "bet365",
            "title": "Bet365",
            "markets": [
                {
                    "key": "btts",
                    "outcomes": [
                        {"name": "Yes", "price": 1.75},
                        {"name": "No", "price": 2.05},
                    ],
                }
            ],
        },
    ],
}

_FAKE_H2H_RESPONSE = {
    "id": "evt_001",
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "bookmakers": [
        {
            "key": "bet365",
            "title": "Bet365",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Arsenal", "price": 1.65},
                        {"name": "Draw", "price": 3.80},
                        {"name": "Chelsea", "price": 5.50},
                    ],
                }
            ],
        },
    ],
}


class TestLiveOddsClient:
    """Tests for the main client class."""

    def _make_client(self, **kwargs):
        """Helper to create a client with a fake API key."""
        defaults = {"api_key": "test_key_123", "cache_ttl_seconds": 900}
        defaults.update(kwargs)
        return LiveOddsClient(**defaults)

    # ----- No API key ----- #

    def test_no_api_key_returns_none(self):
        client = LiveOddsClient(api_key="")
        result = client.get_match_odds("premier_league", "Arsenal", "Chelsea", "cards_under_35")
        assert result is None

    # ----- Quota tracking ----- #

    def test_quota_exhausted_returns_none(self):
        client = self._make_client()
        client.quota.requests_remaining = 0
        result = client.get_match_odds("premier_league", "Arsenal", "Chelsea", "cards_under_35")
        assert result is None

    def test_quota_below_threshold_returns_none(self):
        client = self._make_client(quota_safety_threshold=100)
        client.quota.requests_remaining = 50
        result = client.get_match_odds("premier_league", "Arsenal", "Chelsea", "cards_under_35")
        assert result is None

    def test_quota_updated_from_headers(self):
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {
            "x-requests-remaining": "4500",
            "x-requests-used": "500",
        }
        mock_response.json.return_value = _FAKE_EVENTS_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("src.odds.live_odds_client.http_requests.get", return_value=mock_response):
            client._get_events("soccer_epl")

        assert client.quota.requests_remaining == 4500
        assert client.quota.requests_used == 500
        assert client.quota.session_requests == 1

    # ----- Unknown league ----- #

    def test_unknown_league_returns_none(self):
        client = self._make_client()
        result = client.get_match_odds("unknown_league", "A", "B", "cards_under_35")
        assert result is None

    # ----- Caching ----- #

    def test_cache_hit_skips_api_call(self):
        client = self._make_client()
        # Pre-populate events cache
        client._events_cache["events:soccer_epl"] = CacheEntry(
            data=_FAKE_EVENTS_RESPONSE, timestamp=time.monotonic()
        )
        # Pre-populate odds cache
        client._cache["odds:evt_001:alternate_totals_cards"] = CacheEntry(
            data=_FAKE_TOTALS_RESPONSE, timestamp=time.monotonic()
        )

        with patch("src.odds.live_odds_client.http_requests.get") as mock_get:
            result = client.get_match_odds(
                "premier_league", "Arsenal", "Chelsea", "cards_under_35"
            )
            # No API calls should have been made
            mock_get.assert_not_called()

        assert result is not None
        assert result.line == 3.5

    def test_cache_expired_makes_api_call(self):
        client = self._make_client(cache_ttl_seconds=1)
        # Pre-populate with expired entry
        client._events_cache["events:soccer_epl"] = CacheEntry(
            data=_FAKE_EVENTS_RESPONSE, timestamp=time.monotonic() - 10
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"x-requests-remaining": "100", "x-requests-used": "1"}
        mock_response.json.return_value = _FAKE_EVENTS_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("src.odds.live_odds_client.http_requests.get", return_value=mock_response):
            events = client._get_events("soccer_epl")

        assert len(events) == 2

    def test_clear_cache(self):
        client = self._make_client()
        client._cache["key1"] = CacheEntry(data="a", timestamp=time.monotonic())
        client._events_cache["key2"] = CacheEntry(data="b", timestamp=time.monotonic())
        cleared = client.clear_cache()
        assert cleared == 2
        assert len(client._cache) == 0
        assert len(client._events_cache) == 0

    def test_evict_expired(self):
        client = self._make_client(cache_ttl_seconds=5)
        client._cache["fresh"] = CacheEntry(data="a", timestamp=time.monotonic())
        client._cache["stale"] = CacheEntry(data="b", timestamp=time.monotonic() - 100)
        evicted = client.evict_expired()
        assert evicted == 1
        assert "fresh" in client._cache
        assert "stale" not in client._cache

    # ----- Event matching ----- #

    def test_find_event_exact_match(self):
        client = self._make_client()
        client._events_cache["events:soccer_epl"] = CacheEntry(
            data=_FAKE_EVENTS_RESPONSE, timestamp=time.monotonic()
        )
        event = client._find_event("soccer_epl", "Arsenal", "Chelsea")
        assert event is not None
        assert event["id"] == "evt_001"

    def test_find_event_case_insensitive(self):
        client = self._make_client()
        client._events_cache["events:soccer_epl"] = CacheEntry(
            data=_FAKE_EVENTS_RESPONSE, timestamp=time.monotonic()
        )
        event = client._find_event("soccer_epl", "arsenal", "chelsea")
        assert event is not None
        assert event["id"] == "evt_001"

    def test_find_event_token_overlap(self):
        client = self._make_client()
        events = [
            {
                "id": "evt_003",
                "home_team": "Arsenal FC",
                "away_team": "Chelsea FC",
            }
        ]
        client._events_cache["events:soccer_epl"] = CacheEntry(
            data=events, timestamp=time.monotonic()
        )
        event = client._find_event("soccer_epl", "Arsenal", "Chelsea")
        assert event is not None
        assert event["id"] == "evt_003"

    def test_find_event_no_match(self):
        client = self._make_client()
        client._events_cache["events:soccer_epl"] = CacheEntry(
            data=_FAKE_EVENTS_RESPONSE, timestamp=time.monotonic()
        )
        event = client._find_event("soccer_epl", "Barcelona", "Real Madrid")
        assert event is None

    # ----- Totals parsing (cards, corners, shots, fouls) ----- #

    def test_parse_totals_finds_target_line(self):
        client = self._make_client()
        result = client._parse_totals(
            _FAKE_TOTALS_RESPONSE, "alternate_totals_cards", 3.5
        )
        assert result is not None
        assert result.line == 3.5
        assert result.over_avg is not None
        assert result.under_avg is not None
        # Two bookmakers for line 3.5
        assert result.bookmaker_count == 2

    def test_parse_totals_averages(self):
        client = self._make_client()
        result = client._parse_totals(
            _FAKE_TOTALS_RESPONSE, "alternate_totals_cards", 3.5
        )
        # (1.85 + 1.80) / 2 = 1.825
        assert result.over_avg == pytest.approx(1.825)
        # (1.95 + 2.00) / 2 = 1.975
        assert result.under_avg == pytest.approx(1.975)

    def test_parse_totals_max_odds(self):
        client = self._make_client()
        result = client._parse_totals(
            _FAKE_TOTALS_RESPONSE, "alternate_totals_cards", 3.5
        )
        assert result.over_max == 1.85
        assert result.under_max == 2.00

    def test_parse_totals_available_lines(self):
        client = self._make_client()
        result = client._parse_totals(
            _FAKE_TOTALS_RESPONSE, "alternate_totals_cards", 3.5
        )
        assert 3.5 in result.available_lines
        assert 4.5 in result.available_lines

    def test_parse_totals_rejects_inexact_line(self):
        client = self._make_client()
        # Ask for 4.0 — exact line not available (3.5 and 4.5 exist).
        # Must reject to prevent wrong-line odds creating phantom edges.
        result = client._parse_totals(
            _FAKE_TOTALS_RESPONSE, "alternate_totals_cards", 4.0
        )
        assert result is None

    def test_parse_totals_empty_bookmakers(self):
        client = self._make_client()
        result = client._parse_totals({"bookmakers": []}, "alternate_totals_cards", 3.5)
        assert result is None

    def test_parse_totals_no_bookmakers_key(self):
        client = self._make_client()
        result = client._parse_totals({}, "alternate_totals_cards", 3.5)
        assert result is None

    # ----- BTTS parsing ----- #

    def test_parse_btts(self):
        client = self._make_client()
        result = client._parse_btts(_FAKE_BTTS_RESPONSE)
        assert result is not None
        assert result.over_avg == pytest.approx(1.75)
        assert result.under_avg == pytest.approx(2.05)

    def test_parse_btts_empty(self):
        client = self._make_client()
        result = client._parse_btts({"bookmakers": []})
        assert result is None

    # ----- H2H parsing ----- #

    def test_parse_h2h_home(self):
        client = self._make_client()
        event = {"home_team": "Arsenal", "away_team": "Chelsea"}
        result = client._parse_h2h(_FAKE_H2H_RESPONSE, event, "home")
        assert result is not None
        assert result.over_avg == pytest.approx(1.65)

    def test_parse_h2h_away(self):
        client = self._make_client()
        event = {"home_team": "Arsenal", "away_team": "Chelsea"}
        result = client._parse_h2h(_FAKE_H2H_RESPONSE, event, "away")
        assert result is not None
        assert result.over_avg == pytest.approx(5.50)

    def test_parse_h2h_empty(self):
        client = self._make_client()
        event = {"home_team": "Arsenal", "away_team": "Chelsea"}
        result = client._parse_h2h({"bookmakers": []}, event, "home")
        assert result is None

    # ----- Full integration (mocked API) ----- #

    def test_get_match_odds_cards_under(self):
        """Full path: get_match_odds for cards_under_35."""
        client = self._make_client()

        # Pre-populate events
        client._events_cache["events:soccer_epl"] = CacheEntry(
            data=_FAKE_EVENTS_RESPONSE, timestamp=time.monotonic()
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"x-requests-remaining": "100", "x-requests-used": "1"}
        mock_response.json.return_value = _FAKE_TOTALS_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("src.odds.live_odds_client.http_requests.get", return_value=mock_response):
            result = client.get_match_odds(
                "premier_league", "Arsenal", "Chelsea", "cards_under_35"
            )

        assert result is not None
        assert result.line == 3.5
        assert result.over_avg == pytest.approx(1.825)
        assert result.under_avg == pytest.approx(1.975)
        assert result.source == "the_odds_api"

    def test_get_match_odds_btts(self):
        """Full path: get_match_odds for btts."""
        client = self._make_client()

        client._events_cache["events:soccer_epl"] = CacheEntry(
            data=_FAKE_EVENTS_RESPONSE, timestamp=time.monotonic()
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"x-requests-remaining": "99", "x-requests-used": "2"}
        mock_response.json.return_value = _FAKE_BTTS_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("src.odds.live_odds_client.http_requests.get", return_value=mock_response):
            result = client.get_match_odds(
                "premier_league", "Arsenal", "Chelsea", "btts"
            )

        assert result is not None
        assert result.over_avg == pytest.approx(1.75)
        assert result.under_avg == pytest.approx(2.05)

    def test_get_match_odds_event_not_found(self):
        """Returns None when the match is not in upcoming events."""
        client = self._make_client()
        client._events_cache["events:soccer_epl"] = CacheEntry(
            data=_FAKE_EVENTS_RESPONSE, timestamp=time.monotonic()
        )
        result = client.get_match_odds(
            "premier_league", "Barcelona", "Real Madrid", "cards_under_35"
        )
        assert result is None

    # ----- API error handling ----- #

    def test_api_timeout_returns_none(self):
        client = self._make_client()

        with patch(
            "src.odds.live_odds_client.http_requests.get",
            side_effect=__import__("requests").exceptions.Timeout("timeout"),
        ):
            events = client._get_events("soccer_epl")

        assert events == []

    def test_api_connection_error_returns_none(self):
        client = self._make_client()

        with patch(
            "src.odds.live_odds_client.http_requests.get",
            side_effect=__import__("requests").exceptions.ConnectionError("failed"),
        ):
            events = client._get_events("soccer_epl")

        assert events == []

    def test_api_http_error_returns_none(self):
        client = self._make_client()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = __import__(
            "requests"
        ).exceptions.HTTPError(response=mock_response)

        with patch("src.odds.live_odds_client.http_requests.get", return_value=mock_response):
            result = client._make_request("/test", {})

        assert result is None

    # ----- Batch odds ----- #

    def test_get_batch_odds(self):
        """get_batch_odds returns a dict for each market."""
        client = self._make_client()
        client._events_cache["events:soccer_epl"] = CacheEntry(
            data=_FAKE_EVENTS_RESPONSE, timestamp=time.monotonic()
        )
        client._cache["odds:evt_001:alternate_totals_cards"] = CacheEntry(
            data=_FAKE_TOTALS_RESPONSE, timestamp=time.monotonic()
        )
        client._cache["odds:evt_001:btts"] = CacheEntry(
            data=_FAKE_BTTS_RESPONSE, timestamp=time.monotonic()
        )

        with patch("src.odds.live_odds_client.http_requests.get") as mock_get:
            results = client.get_batch_odds(
                "premier_league",
                "Arsenal",
                "Chelsea",
                ["cards_under_35", "btts"],
            )
            mock_get.assert_not_called()

        assert "cards_under_35" in results
        assert "btts" in results
        assert results["cards_under_35"] is not None
        assert results["btts"] is not None

    # ----- MatchOdds dataclass ----- #

    def test_match_odds_defaults(self):
        odds = MatchOdds()
        assert odds.over_avg is None
        assert odds.under_avg is None
        assert odds.line is None
        assert odds.available_lines == []
        assert odds.bookmaker_count == 0
        assert odds.source == "the_odds_api"

    def test_match_odds_with_values(self):
        odds = MatchOdds(
            over_avg=1.85,
            under_avg=1.95,
            line=3.5,
            available_lines=[3.5, 4.5, 5.5],
            bookmaker_count=5,
        )
        assert odds.over_avg == 1.85
        assert odds.line == 3.5
        assert len(odds.available_lines) == 3

    # ----- get_supported_markets ----- #

    def test_get_supported_markets(self):
        supported = LiveOddsClient.get_supported_markets()
        assert "cards" in supported
        assert "corners" in supported
        assert "shots" in supported
        assert "btts" in supported
        assert "fouls" not in supported  # No API market exists

    # ----- Fouls market not supported ----- #

    def test_fouls_market_returns_none(self):
        """Fouls has no API market — should return None gracefully."""
        client = self._make_client()
        client._events_cache["events:soccer_epl"] = CacheEntry(
            data=_FAKE_EVENTS_RESPONSE, timestamp=time.monotonic()
        )
        result = client.get_match_odds(
            "premier_league", "Arsenal", "Chelsea", "fouls_under_255"
        )
        assert result is None

    # ----- JSON decode error ----- #

    def test_json_decode_error_returns_none(self):
        """Invalid JSON response should return None, not crash."""
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"x-requests-remaining": "100", "x-requests-used": "1"}
        mock_response.json.side_effect = ValueError("No JSON")
        mock_response.raise_for_status = MagicMock()

        with patch("src.odds.live_odds_client.http_requests.get", return_value=mock_response):
            result = client._make_request("/test", {})
        assert result is None

    # ----- Token overlap matching ----- #

    def test_find_event_no_overlap(self):
        """Completely different team names should not match."""
        client = self._make_client()
        events = [{"id": "evt_010", "home_team": "Manchester City", "away_team": "Tottenham Hotspur"}]
        client._events_cache["events:soccer_epl"] = CacheEntry(
            data=events, timestamp=time.monotonic()
        )
        event = client._find_event("soccer_epl", "Arsenal", "Chelsea")
        assert event is None

    def test_find_event_picks_best_overlap(self):
        """When multiple events match, should pick the best overlap score."""
        client = self._make_client()
        events = [
            {"id": "evt_010", "home_team": "AC Milan", "away_team": "Como 1907"},
            {"id": "evt_011", "home_team": "Milan Primavera", "away_team": "Como Youth"},
        ]
        client._events_cache["events:soccer_italy_serie_a"] = CacheEntry(
            data=events, timestamp=time.monotonic()
        )
        event = client._find_event("soccer_italy_serie_a", "AC Milan", "Como")
        assert event is not None
        assert event["id"] == "evt_010"


class TestToPipelineOdds:
    """Tests for the pipeline integration adapter."""

    def test_niche_cards_under(self):
        odds = MatchOdds(over_avg=1.85, under_avg=1.95, line=3.5)
        result = to_pipeline_odds("cards_under_35", odds)
        assert result["cards_over_avg"] == 1.85
        assert result["cards_under_avg"] == 1.95

    def test_niche_corners_over(self):
        odds = MatchOdds(over_avg=2.10, under_avg=1.70, line=9.5)
        result = to_pipeline_odds("corners_over_95", odds)
        assert result["corners_over_avg"] == 2.10
        assert result["corners_under_avg"] == 1.70

    def test_btts(self):
        odds = MatchOdds(over_avg=1.75, under_avg=2.05)
        result = to_pipeline_odds("btts", odds)
        assert result["btts_yes_avg"] == 1.75
        assert result["btts_no_avg"] == 2.05

    def test_home_win(self):
        odds = MatchOdds(over_avg=1.65, under_avg=3.80)
        result = to_pipeline_odds("home_win", odds)
        assert result["h2h_home_avg"] == 1.65
        assert result["h2h_draw_avg"] == 3.80

    def test_over25(self):
        odds = MatchOdds(over_avg=1.90, under_avg=1.90)
        result = to_pipeline_odds("over25", odds)
        assert result["totals_over_avg"] == 1.90
        assert result["totals_under_avg"] == 1.90

    def test_none_values_excluded(self):
        odds = MatchOdds(over_avg=1.85, under_avg=None)
        result = to_pipeline_odds("cards_under_35", odds)
        assert "cards_over_avg" in result
        assert "cards_under_avg" not in result

    def test_shots_market(self):
        odds = MatchOdds(over_avg=1.95, under_avg=1.85, line=26.5)
        result = to_pipeline_odds("shots_over_265", odds)
        assert result["shots_over_avg"] == 1.95
        assert result["shots_under_avg"] == 1.85

    # ----- Per-line column output (F9 fix) ----- #

    def test_niche_cards_per_line_columns(self):
        """Line-specific markets produce per-line columns alongside generic ones."""
        odds = MatchOdds(over_avg=1.85, under_avg=1.95, line=3.5)
        result = to_pipeline_odds("cards_under_35", odds)
        # Generic columns (backward compat)
        assert result["cards_over_avg"] == 1.85
        assert result["cards_under_avg"] == 1.95
        # Per-line columns (new)
        assert result["cards_over_avg_35"] == 1.85
        assert result["cards_under_avg_35"] == 1.95

    def test_niche_corners_per_line_columns(self):
        """Corners over 8.5 produces corners_over_avg_85 column."""
        odds = MatchOdds(over_avg=1.50, under_avg=2.60, line=8.5)
        result = to_pipeline_odds("corners_over_85", odds)
        assert result["corners_over_avg_85"] == 1.50
        assert result["corners_under_avg_85"] == 2.60

    def test_niche_shots_per_line_columns(self):
        """Shots over 28.5 produces shots_over_avg_285 column."""
        odds = MatchOdds(over_avg=2.10, under_avg=1.70, line=28.5)
        result = to_pipeline_odds("shots_over_285", odds)
        assert result["shots_over_avg_285"] == 2.10
        assert result["shots_under_avg_285"] == 1.70

    def test_base_market_no_per_line_columns(self):
        """Base markets (no line) should NOT produce per-line columns."""
        odds = MatchOdds(over_avg=1.95, under_avg=1.85)
        result = to_pipeline_odds("shots", odds)
        assert result["shots_over_avg"] == 1.95
        assert result["shots_under_avg"] == 1.85
        # No per-line columns for base markets
        per_line_keys = [k for k in result if k.endswith(("_245", "_255", "_265"))]
        assert per_line_keys == []

    def test_h2h_no_per_line_columns(self):
        """H2H markets should NOT produce per-line columns."""
        odds = MatchOdds(over_avg=1.65, under_avg=3.80)
        result = to_pipeline_odds("home_win", odds)
        assert result == {"h2h_home_avg": 1.65, "h2h_draw_avg": 3.80}

    def test_per_line_none_values_excluded(self):
        """None values should be excluded from per-line columns too."""
        odds = MatchOdds(over_avg=1.85, under_avg=None, line=3.5)
        result = to_pipeline_odds("cards_over_35", odds)
        assert "cards_over_avg_35" in result
        assert "cards_under_avg_35" not in result
