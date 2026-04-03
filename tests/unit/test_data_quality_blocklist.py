"""Tests for data quality blocklist, EMA floor dampening, and inactive leagues."""

import numpy as np
import pandas as pd
import pytest

from src.data_quality import (
    fix_corrupted_odds,
    fix_fake_zero_cards,
    fix_fake_zero_stats,
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
        assert "turkish_super_lig" in blocklist["cards"]

    def test_load_blocklist_missing_file(self):
        blocklist = load_blocklist("nonexistent.yaml")
        assert blocklist == {}

    def test_cards_has_expected_leagues(self):
        blocklist = load_blocklist("config/strategies.yaml")
        assert set(blocklist["cards"]) == {
            "turkish_super_lig",
            "scottish_premiership",
            "belgian_pro_league",
            "ligue_1",
            "la_liga_2",
            "championship",
            "eredivisie",
            "ekstraklasa",
            "la_liga",
        }

    def test_fouls_has_expected_leagues(self):
        blocklist = load_blocklist("config/strategies.yaml")
        assert set(blocklist["fouls"]) == {
            "belgian_pro_league",
            "la_liga_2",
            "championship",
        }

    def test_corners_has_expected_leagues(self):
        blocklist = load_blocklist("config/strategies.yaml")
        assert set(blocklist["corners"]) == {
            "belgian_pro_league",
            "la_liga_2",
            "championship",
        }

    def test_shots_has_expected_leagues(self):
        blocklist = load_blocklist("config/strategies.yaml")
        assert set(blocklist["shots"]) == {
            "belgian_pro_league",
            "la_liga_2",
            "championship",
        }


class TestIsMarketBlocked:
    """Test market blocking logic."""

    @pytest.fixture
    def blocklist(self):
        return load_blocklist("config/strategies.yaml")

    def test_cards_under_25_blocked_turkish(self, blocklist):
        assert is_market_blocked_for_league("cards_under_25", "turkish_super_lig", blocklist)

    def test_cards_over_35_blocked_scottish(self, blocklist):
        assert is_market_blocked_for_league("cards_over_35", "scottish_premiership", blocklist)

    def test_cardshc_blocked_for_cards_leagues(self, blocklist):
        assert is_market_blocked_for_league("cardshc_under_05", "turkish_super_lig", blocklist)

    def test_cards_blocked_for_eredivisie(self, blocklist):
        assert is_market_blocked_for_league("cards_under_25", "eredivisie", blocklist)

    def test_cards_allowed_for_serie_a(self, blocklist):
        assert not is_market_blocked_for_league("cards_over_35", "serie_a", blocklist)

    def test_cards_blocked_for_la_liga(self, blocklist):
        assert is_market_blocked_for_league("cards_under_25", "la_liga", blocklist)

    def test_fouls_blocked_for_belgian(self, blocklist):
        assert is_market_blocked_for_league("fouls_over_225", "belgian_pro_league", blocklist)

    def test_fouls_allowed_for_bundesliga(self, blocklist):
        assert not is_market_blocked_for_league("fouls_over_225", "bundesliga", blocklist)

    def test_corners_blocked_for_championship(self, blocklist):
        assert is_market_blocked_for_league("corners_over_85", "championship", blocklist)

    def test_shots_blocked_for_la_liga_2(self, blocklist):
        assert is_market_blocked_for_league("shots_over_255", "la_liga_2", blocklist)

    def test_home_win_not_blocked(self, blocklist):
        assert not is_market_blocked_for_league("home_win", "eredivisie", blocklist)

    def test_btts_not_blocked(self, blocklist):
        assert not is_market_blocked_for_league("btts", "turkish_super_lig", blocklist)

    def test_empty_blocklist_allows_all(self):
        assert not is_market_blocked_for_league("cards_under_25", "eredivisie", {})

    def test_none_blocklist_loads_default(self):
        # When None is passed, loads from default config
        result = is_market_blocked_for_league("cards_under_25", "turkish_super_lig", None)
        assert result is True

    def test_shots_blocked_for_championship(self, blocklist):
        assert is_market_blocked_for_league("shots_over_245", "championship", blocklist)

    def test_shots_blocked_for_belgian(self, blocklist):
        assert is_market_blocked_for_league("shots_under_285", "belgian_pro_league", blocklist)

    def test_fouls_blocked_for_la_liga_2(self, blocklist):
        assert is_market_blocked_for_league("fouls_over_225", "la_liga_2", blocklist)


class TestInactiveLeagues:
    """Test inactive league loading."""

    def test_load_inactive_leagues(self):
        inactive = load_inactive_leagues()
        assert set(inactive) == set()

    def test_load_inactive_leagues_missing_file(self):
        inactive = load_inactive_leagues("nonexistent.yaml")
        assert inactive == []


class TestFixFakeZeroCards:
    """Test fake zero card detection and NaN-ification."""

    def _make_df(self, home_cards, away_cards, home_fouls, away_fouls,
                 home_yc=None, away_yc=None):
        df = pd.DataFrame({
            "home_cards": home_cards,
            "away_cards": away_cards,
            "home_fouls": home_fouls,
            "away_fouls": away_fouls,
            "home_yellow_cards": home_yc if home_yc is not None else home_cards,
            "away_yellow_cards": away_yc if away_yc is not None else away_cards,
            "total_cards": [h + a for h, a in zip(home_cards, away_cards)],
        })
        return df

    def test_both_zero_with_fouls_becomes_nan(self):
        df = self._make_df([0, 2], [0, 3], [12, 10], [14, 8])
        result = fix_fake_zero_cards(df)
        assert np.isnan(result.loc[0, "home_cards"])
        assert np.isnan(result.loc[0, "total_cards"])
        assert result.loc[1, "home_cards"] == 2  # Real data untouched

    def test_both_zero_low_fouls_kept(self):
        """Rare real zero-card match with low fouls stays as 0."""
        df = self._make_df([0], [0], [2], [1])
        result = fix_fake_zero_cards(df)
        assert result.loc[0, "home_cards"] == 0
        assert result.loc[0, "total_cards"] == 0

    def test_home_side_fake_zero(self):
        """Home yellows=0 with high home fouls -> home side NaN."""
        df = self._make_df([0, 3], [2, 1], [15, 10], [8, 5],
                           home_yc=[0, 2], away_yc=[2, 1])
        result = fix_fake_zero_cards(df)
        assert np.isnan(result.loc[0, "home_cards"])
        assert result.loc[0, "away_cards"] == 2  # Away side untouched

    def test_no_fouls_column_returns_unchanged(self):
        df = pd.DataFrame({"home_cards": [0, 3], "away_cards": [0, 1]})
        result = fix_fake_zero_cards(df)
        assert result.loc[0, "home_cards"] == 0

    def test_total_cards_recomputed(self):
        df = self._make_df([0, 2], [0, 3], [12, 10], [14, 8])
        result = fix_fake_zero_cards(df)
        assert np.isnan(result.loc[0, "total_cards"])
        assert result.loc[1, "total_cards"] == 5

    def test_away_only_fake_zero(self):
        """away_cards=0 with home_cards>0 → away_cards becomes NaN."""
        df = self._make_df([2, 3], [0, 1], [10, 12], [8, 9])
        result = fix_fake_zero_cards(df)
        assert np.isnan(result.loc[0, "away_cards"])
        assert result.loc[0, "home_cards"] == 2  # Home untouched
        assert result.loc[1, "away_cards"] == 1  # Real data untouched


class TestFixFakeZeroStats:
    def test_fake_zero_shots_with_goals(self):
        """Matches with goals but 0 shots → NaN."""
        df = pd.DataFrame({
            "total_goals": [3, 0, 2],
            "total_shots": [0, 0, 15],
            "home_shots": [0, 0, 8],
            "away_shots": [0, 0, 7],
            "home_shots_on_target": [0, 0, 4],
            "away_shots_on_target": [0, 0, 3],
            "total_shots_on_target": [0, 0, 7],
        })
        result = fix_fake_zero_stats(df)
        # Row 0: goals=3 + shots=0 → fake → NaN
        assert np.isnan(result.loc[0, "total_shots"])
        assert np.isnan(result.loc[0, "home_shots"])
        # Row 1: goals=0 + shots=0 → could be real 0-0 → keep
        assert result.loc[1, "total_shots"] == 0
        # Row 2: normal data → untouched
        assert result.loc[2, "total_shots"] == 15

    def test_fake_zero_fouls(self):
        """Both fouls=0 with goals → NaN."""
        df = pd.DataFrame({
            "total_goals": [2, 1],
            "home_fouls": [0, 5],
            "away_fouls": [0, 8],
            "total_fouls": [0, 13],
        })
        result = fix_fake_zero_stats(df)
        assert np.isnan(result.loc[0, "home_fouls"])
        assert result.loc[1, "home_fouls"] == 5

    def test_fake_away_corners(self):
        """away_corners=0 when total==home → fake zero."""
        df = pd.DataFrame({
            "home_corners": [5, 3],
            "away_corners": [0, 4],
            "total_corners": [5, 7],
        })
        result = fix_fake_zero_stats(df)
        assert np.isnan(result.loc[0, "away_corners"])
        assert np.isnan(result.loc[0, "total_corners"])
        assert result.loc[1, "away_corners"] == 4

    def test_no_columns_returns_unchanged(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = fix_fake_zero_stats(df)
        assert len(result) == 3


class TestFixCorruptedOdds:
    def test_low_overround_nan(self):
        """Overround < 0.90 → NaN avg/max odds."""
        df = pd.DataFrame({
            "avg_home_close": [3.59, 2.0],
            "avg_draw_close": [3.88, 3.5],
            "avg_away_close": [5.13, 3.0],
            "max_home_close": [41.0, 2.1],
            "odds_home_prob": [0.28, 0.50],
        })
        result = fix_corrupted_odds(df)
        # Row 0: overround = 1/3.59 + 1/3.88 + 1/5.13 ≈ 0.73 → corrupted
        assert np.isnan(result.loc[0, "avg_home_close"])
        assert np.isnan(result.loc[0, "max_home_close"])
        assert np.isnan(result.loc[0, "odds_home_prob"])
        # Row 1: overround = 1/2 + 1/3.5 + 1/3 ≈ 1.12 → OK
        assert result.loc[1, "avg_home_close"] == 2.0

    def test_no_odds_returns_unchanged(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = fix_corrupted_odds(df)
        assert len(result) == 2
