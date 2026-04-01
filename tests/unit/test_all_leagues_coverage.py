"""Tests that ALL_LEAGUES covers every league in LEAGUE_IDS."""
import pytest
from src.leagues import LEAGUE_IDS, ALL_LEAGUES, EUROPEAN_LEAGUES, AMERICAS_LEAGUES


class TestAllLeagues:
    def test_all_leagues_covers_every_league_id(self):
        """ALL_LEAGUES must include every key in LEAGUE_IDS."""
        assert set(ALL_LEAGUES) == set(LEAGUE_IDS.keys())

    def test_all_leagues_is_european_plus_americas(self):
        """ALL_LEAGUES = EUROPEAN_LEAGUES + AMERICAS_LEAGUES."""
        assert set(ALL_LEAGUES) == set(EUROPEAN_LEAGUES) | set(AMERICAS_LEAGUES)

    def test_no_duplicates_in_all_leagues(self):
        assert len(ALL_LEAGUES) == len(set(ALL_LEAGUES))
