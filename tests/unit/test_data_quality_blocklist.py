"""Tests for data quality blocklist, EMA floor dampening, and inactive leagues."""

import pytest

from src.data_quality import (
    get_base_market,
    is_market_blocked_for_league,
    load_blocklist,
    load_inactive_leagues,
)


class TestGetBaseMarket:
    """Test base market extraction from market names."""

    def test_cards_under_25(self):
        assert get_base_market("cards_under_25") == "cards"

    def test_cards_over_35(self):
        assert get_base_market("cards_over_35") == "cards"

    def test_cardshc_under_05(self):
        assert get_base_market("cardshc_under_05") == "cards"

    def test_fouls_over_225(self):
        assert get_base_market("fouls_over_225") == "fouls"

    def test_corners_over_85(self):
        assert get_base_market("corners_over_85") == "corners"

    def test_cornershc_over_05(self):
        assert get_base_market("cornershc_over_05") == "corners"

    def test_shots_under_255(self):
        assert get_base_market("shots_under_255") == "shots"

    def test_over25_is_goals(self):
        assert get_base_market("over25") == "goals"

    def test_under25_is_goals(self):
        assert get_base_market("under25") == "goals"

    def test_home_win_is_h2h(self):
        assert get_base_market("home_win") == "h2h"

    def test_away_win_is_h2h(self):
        assert get_base_market("away_win") == "h2h"

    def test_btts(self):
        assert get_base_market("btts") == "btts"

    def test_ht_markets(self):
        assert get_base_market("ht_over_05") == "ht"


class TestLoadBlocklist:
    """Test blocklist loading from strategies.yaml."""

    def test_load_blocklist_from_yaml(self):
        blocklist = load_blocklist("config/strategies.yaml")
        assert isinstance(blocklist, dict)
        assert "cards" in blocklist
        assert "eredivisie" in blocklist["cards"]

    def test_load_blocklist_missing_file(self):
        blocklist = load_blocklist("nonexistent.yaml")
        assert blocklist == {}

    def test_cards_has_four_leagues(self):
        blocklist = load_blocklist("config/strategies.yaml")
        assert set(blocklist["cards"]) == {
            "eredivisie",
            "turkish_super_lig",
            "serie_a",
            "scottish_premiership",
        }

    def test_fouls_has_turkish_and_ligue1(self):
        blocklist = load_blocklist("config/strategies.yaml")
        assert set(blocklist["fouls"]) == {"turkish_super_lig", "ligue_1"}

    def test_corners_has_two_leagues(self):
        blocklist = load_blocklist("config/strategies.yaml")
        assert set(blocklist["corners"]) == {"ligue_1", "belgian_pro_league"}

    def test_shots_has_two_leagues(self):
        blocklist = load_blocklist("config/strategies.yaml")
        assert set(blocklist["shots"]) == {"ligue_1", "belgian_pro_league"}


class TestIsMarketBlocked:
    """Test market blocking logic."""

    @pytest.fixture
    def blocklist(self):
        return load_blocklist("config/strategies.yaml")

    def test_cards_under_25_blocked_eredivisie(self, blocklist):
        assert is_market_blocked_for_league("cards_under_25", "eredivisie", blocklist)

    def test_cards_over_35_blocked_serie_a(self, blocklist):
        assert is_market_blocked_for_league("cards_over_35", "serie_a", blocklist)

    def test_cardshc_blocked_for_cards_leagues(self, blocklist):
        assert is_market_blocked_for_league("cardshc_under_05", "eredivisie", blocklist)

    def test_cards_allowed_for_la_liga(self, blocklist):
        assert not is_market_blocked_for_league("cards_under_25", "la_liga", blocklist)

    def test_fouls_blocked_for_turkish(self, blocklist):
        assert is_market_blocked_for_league("fouls_over_225", "turkish_super_lig", blocklist)

    def test_fouls_allowed_for_bundesliga(self, blocklist):
        assert not is_market_blocked_for_league("fouls_over_225", "bundesliga", blocklist)

    def test_corners_blocked_for_ligue1(self, blocklist):
        assert is_market_blocked_for_league("corners_over_85", "ligue_1", blocklist)

    def test_shots_not_blocked_anywhere(self, blocklist):
        assert not is_market_blocked_for_league("shots_over_255", "eredivisie", blocklist)

    def test_home_win_not_blocked(self, blocklist):
        assert not is_market_blocked_for_league("home_win", "eredivisie", blocklist)

    def test_btts_not_blocked(self, blocklist):
        assert not is_market_blocked_for_league("btts", "turkish_super_lig", blocklist)

    def test_empty_blocklist_allows_all(self):
        assert not is_market_blocked_for_league("cards_under_25", "eredivisie", {})

    def test_none_blocklist_loads_default(self):
        # When None is passed, loads from default config
        result = is_market_blocked_for_league("cards_under_25", "eredivisie", None)
        assert result is True

    def test_shots_blocked_for_ligue1(self, blocklist):
        assert is_market_blocked_for_league("shots_over_245", "ligue_1", blocklist)

    def test_shots_blocked_for_belgian(self, blocklist):
        assert is_market_blocked_for_league("shots_under_285", "belgian_pro_league", blocklist)

    def test_fouls_blocked_for_ligue1(self, blocklist):
        assert is_market_blocked_for_league("fouls_over_225", "ligue_1", blocklist)


class TestInactiveLeagues:
    """Test inactive league loading."""

    def test_load_inactive_leagues(self):
        inactive = load_inactive_leagues()
        assert set(inactive) == {"liga_mx", "mls", "ekstraklasa"}

    def test_load_inactive_leagues_missing_file(self):
        inactive = load_inactive_leagues("nonexistent.yaml")
        assert inactive == []
