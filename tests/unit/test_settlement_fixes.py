"""Tests for settlement bug fixes: cards columns, side/bet_type fallback, shots column name, strategy scores cards."""
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def matches_df():
    """Minimal matches DataFrame for settlement tests."""
    return pd.DataFrame({
        "fixture.id": [100, 101, 102],
        "fixture.status.short": ["FT", "FT", "FT"],
        "fixture.date": ["2026-01-17", "2026-01-17", "2026-01-17"],
        "goals.home": [2, 1, 0],
        "goals.away": [1, 1, 3],
        "teams.home.name": ["Chelsea", "Arsenal", "Liverpool"],
        "teams.away.name": ["Brentford", "Spurs", "Wolves"],
    })


@pytest.fixture
def stats_df_with_cards():
    """Match stats DataFrame WITH yellow/red cards columns (post-fix)."""
    return pd.DataFrame({
        "fixture_id": [100, 101, 102],
        "home_corners": [3, 6, 8],
        "away_corners": [9, 4, 5],
        "home_shots": [6, 12, 8],
        "away_shots": [15, 9, 14],
        "home_fouls": [7, 10, 12],
        "away_fouls": [10, 8, 9],
        "home_yellow_cards": [2, 3, 1],
        "away_yellow_cards": [3, 2, 4],
        "home_red_cards": [0, 0, 1],
        "away_red_cards": [0, 1, 0],
    })


@pytest.fixture
def stats_df_without_cards():
    """Match stats DataFrame WITHOUT cards columns (pre-fix / legacy)."""
    return pd.DataFrame({
        "fixture_id": [100, 101, 102],
        "home_corners": [3, 6, 8],
        "away_corners": [9, 4, 5],
        "home_shots": [6, 12, 8],
        "away_shots": [15, 9, 14],
        "home_fouls": [7, 10, 12],
        "away_fouls": [10, 8, 9],
    })


@pytest.fixture
def stats_df_old_shots():
    """Match stats with old column name home_shots_total."""
    return pd.DataFrame({
        "fixture_id": [100],
        "home_shots_total": [6],
        "away_shots_total": [15],
        "home_fouls": [7],
        "away_fouls": [10],
        "home_corners": [3],
        "away_corners": [9],
    })


# ---------------------------------------------------------------------------
# 1. Cards settlement: columns present → settle correctly
# ---------------------------------------------------------------------------
class TestCardsSettlement:
    """Test that cards bets settle correctly when yellow_cards columns exist."""

    def test_cards_over_won(self, matches_df, stats_df_with_cards):
        from experiments.update_results import settle_recommendation

        row = pd.Series({
            "fixture_id": 100,
            "market": "CARDS",
            "side": "OVER",
            "bet_type": "OVER",
            "line": 4.5,
        })
        result = settle_recommendation(row, matches_df, stats_df_with_cards)
        assert result is not None
        # 2 + 3 = 5 total cards, > 4.5 → WON
        assert result["won"] == True
        assert result["actual_value"] == 5

    def test_cards_over_lost(self, matches_df, stats_df_with_cards):
        from experiments.update_results import settle_recommendation

        row = pd.Series({
            "fixture_id": 100,
            "market": "CARDS",
            "side": "OVER",
            "bet_type": "OVER",
            "line": 5.5,
        })
        result = settle_recommendation(row, matches_df, stats_df_with_cards)
        assert result is not None
        # 2 + 3 = 5 total cards, not > 5.5 → LOST
        assert result["won"] == False

    def test_cards_under_won(self, matches_df, stats_df_with_cards):
        from experiments.update_results import settle_recommendation

        row = pd.Series({
            "fixture_id": 100,
            "market": "CARDS",
            "side": "UNDER",
            "bet_type": "UNDER",
            "line": 5.5,
        })
        result = settle_recommendation(row, matches_df, stats_df_with_cards)
        assert result is not None
        # 5 < 5.5 → WON
        assert result["won"] == True

    def test_cards_returns_none_without_columns(self, matches_df, stats_df_without_cards):
        """When match_stats lacks yellow_cards columns, return None (not False)."""
        from experiments.update_results import settle_recommendation

        row = pd.Series({
            "fixture_id": 100,
            "market": "CARDS",
            "side": "OVER",
            "bet_type": "OVER",
            "line": 3.5,
        })
        result = settle_recommendation(row, matches_df, stats_df_without_cards)
        assert result is None


# ---------------------------------------------------------------------------
# 2. Side vs bet_type fallback
# ---------------------------------------------------------------------------
class TestSideBetTypeFallback:
    """Test that settlement falls back to bet_type when side is missing."""

    def test_side_present_used(self, matches_df, stats_df_with_cards):
        from experiments.update_results import settle_recommendation

        row = pd.Series({
            "fixture_id": 100,
            "market": "CARDS",
            "side": "OVER",
            "bet_type": "UNDER",  # conflicting — side should win
            "line": 4.5,
        })
        result = settle_recommendation(row, matches_df, stats_df_with_cards)
        assert result is not None
        # side=OVER: 5 > 4.5 → WON
        assert result["won"] == True

    def test_side_nan_falls_back_to_bet_type(self, matches_df, stats_df_with_cards):
        from experiments.update_results import settle_recommendation

        row = pd.Series({
            "fixture_id": 100,
            "market": "CARDS",
            "side": None,
            "bet_type": "OVER",
            "line": 4.5,
        })
        result = settle_recommendation(row, matches_df, stats_df_with_cards)
        assert result is not None
        # bet_type=OVER: 5 > 4.5 → WON
        assert result["won"] == True

    def test_side_missing_falls_back_to_bet_type(self, matches_df, stats_df_with_cards):
        from experiments.update_results import settle_recommendation

        # Simulate old CSV format without 'side' key at all
        row = pd.Series({
            "fixture_id": 100,
            "market": "CARDS",
            "bet_type": "UNDER",
            "line": 4.5,
        })
        result = settle_recommendation(row, matches_df, stats_df_with_cards)
        assert result is not None
        # bet_type=UNDER: 5 < 4.5 is False → LOST
        assert result["won"] == False

    def test_corners_over_with_bet_type(self, matches_df, stats_df_with_cards):
        """Verify OVER for corners also works via bet_type fallback."""
        from experiments.update_results import settle_recommendation

        row = pd.Series({
            "fixture_id": 100,
            "market": "CORNERS",
            "side": None,
            "bet_type": "OVER",
            "line": 10.5,
        })
        result = settle_recommendation(row, matches_df, stats_df_with_cards)
        assert result is not None
        # 3 + 9 = 12 > 10.5 → WON
        assert result["won"] == True
        assert result["actual_value"] == 12


# ---------------------------------------------------------------------------
# 3. Shots column name fallback
# ---------------------------------------------------------------------------
class TestShotsColumnFallback:
    """Test that shots settlement works with both old and new column names."""

    def test_shots_new_column_name(self, matches_df, stats_df_with_cards):
        """home_shots / away_shots (current format)."""
        from experiments.update_results import settle_recommendation

        row = pd.Series({
            "fixture_id": 100,
            "market": "SHOTS",
            "side": "OVER",
            "bet_type": "OVER",
            "line": 20.5,
        })
        result = settle_recommendation(row, matches_df, stats_df_with_cards)
        assert result is not None
        # 6 + 15 = 21 > 20.5 → WON
        assert result["won"] == True
        assert result["actual_value"] == 21

    def test_shots_old_column_name(self, matches_df, stats_df_old_shots):
        """home_shots_total / away_shots_total (legacy format)."""
        from experiments.update_results import settle_recommendation

        row = pd.Series({
            "fixture_id": 100,
            "market": "SHOTS",
            "side": "OVER",
            "bet_type": "OVER",
            "line": 20.5,
        })
        result = settle_recommendation(row, matches_df, stats_df_old_shots)
        assert result is not None
        # 6 + 15 = 21 > 20.5 → WON
        assert result["won"] == True

    def test_shots_missing_columns_returns_none(self, matches_df):
        """If neither column variant exists, return None."""
        from experiments.update_results import settle_recommendation

        stats_no_shots = pd.DataFrame({
            "fixture_id": [100],
            "home_fouls": [7],
            "away_fouls": [10],
        })
        row = pd.Series({
            "fixture_id": 100,
            "market": "SHOTS",
            "side": "OVER",
            "bet_type": "OVER",
            "line": 20.5,
        })
        result = settle_recommendation(row, matches_df, stats_no_shots)
        assert result is None


# ---------------------------------------------------------------------------
# 4. Strategy scores cards handler
# ---------------------------------------------------------------------------
class TestStrategyScoresCards:
    """Test that evaluate_market handles cards markets."""

    def test_cards_over_won(self):
        from experiments.update_strategy_scores import evaluate_market

        fixture = {"status": "FT", "home_goals": 2, "away_goals": 1}
        stats = {
            "home": {"yellow_cards": 3, "fouls": 10},
            "away": {"yellow_cards": 2, "fouls": 8},
        }
        result = evaluate_market("cards", fixture, stats, line=4.5)
        assert result is not None
        won, actual = result
        # 3 + 2 = 5 > 4.5 → True
        assert won == True
        assert actual == 5.0

    def test_cards_over_lost(self):
        from experiments.update_strategy_scores import evaluate_market

        fixture = {"status": "FT", "home_goals": 1, "away_goals": 0}
        stats = {
            "home": {"yellow_cards": 1, "fouls": 5},
            "away": {"yellow_cards": 1, "fouls": 7},
        }
        result = evaluate_market("cards", fixture, stats, line=3.5)
        assert result is not None
        won, actual = result
        # 1 + 1 = 2, not > 3.5
        assert won == False
        assert actual == 2.0

    def test_cards_over_35_variant(self):
        """cards_over_35 market name should also match."""
        from experiments.update_strategy_scores import evaluate_market

        fixture = {"status": "FT", "home_goals": 0, "away_goals": 0}
        stats = {
            "home": {"yellow_cards": 2},
            "away": {"yellow_cards": 3},
        }
        result = evaluate_market("cards_over_35", fixture, stats, line=3.5)
        assert result is not None
        won, actual = result
        # 2 + 3 = 5 > 3.5
        assert won == True

    def test_cards_returns_none_without_stats(self):
        from experiments.update_strategy_scores import evaluate_market

        fixture = {"status": "FT", "home_goals": 1, "away_goals": 1}
        result = evaluate_market("cards", fixture, None, line=3.5)
        assert result is None

    def test_cards_returns_none_if_not_finished(self):
        from experiments.update_strategy_scores import evaluate_market

        fixture = {"status": "NS", "home_goals": None, "away_goals": None}
        stats = {"home": {"yellow_cards": 2}, "away": {"yellow_cards": 1}}
        result = evaluate_market("cards", fixture, stats, line=3.5)
        assert result is None


# ---------------------------------------------------------------------------
# 5. collect_all_stats record includes cards columns
# ---------------------------------------------------------------------------
class TestCollectAllStatsRecord:
    """Test that collect_all_stats builds records with cards columns."""

    def test_record_includes_yellow_and_red_cards(self):
        from scripts.collect_all_stats import clean_val

        # Simulate what collect_league_season builds from API response
        home_stats = {
            "Corner Kicks": 5,
            "Total Shots": 12,
            "Shots on Goal": 4,
            "Fouls": 10,
            "Yellow Cards": 3,
            "Red Cards": 0,
            "Ball Possession": "55%",
            "Offsides": 2,
        }
        away_stats = {
            "Corner Kicks": 7,
            "Total Shots": 9,
            "Shots on Goal": 3,
            "Fouls": 8,
            "Yellow Cards": 2,
            "Red Cards": 1,
            "Ball Possession": "45%",
            "Offsides": 1,
        }

        record = {
            "fixture_id": 999,
            "home_yellow_cards": clean_val(home_stats.get("Yellow Cards")),
            "away_yellow_cards": clean_val(away_stats.get("Yellow Cards")),
            "home_red_cards": clean_val(home_stats.get("Red Cards")),
            "away_red_cards": clean_val(away_stats.get("Red Cards")),
        }

        assert record["home_yellow_cards"] == 3
        assert record["away_yellow_cards"] == 2
        assert record["home_red_cards"] == 0
        assert record["away_red_cards"] == 1

    def test_clean_val_handles_none(self):
        from scripts.collect_all_stats import clean_val

        assert clean_val(None) == 0
        assert clean_val(None, default=5) == 5

    def test_clean_val_handles_string_number(self):
        from scripts.collect_all_stats import clean_val

        assert clean_val("3") == 3
        assert clean_val("0") == 0

    def test_clean_val_handles_percentage(self):
        from scripts.collect_all_stats import clean_val

        assert clean_val("55%") == 55


# ---------------------------------------------------------------------------
# 6. Fouls and corners still work (regression check)
# ---------------------------------------------------------------------------
class TestNicheMarketsRegression:
    """Verify fouls and corners settlement isn't broken by the fixes."""

    def test_fouls_over(self, matches_df, stats_df_with_cards):
        from experiments.update_results import settle_recommendation

        row = pd.Series({
            "fixture_id": 100,
            "market": "FOULS",
            "side": "OVER",
            "bet_type": "OVER",
            "line": 16.5,
        })
        result = settle_recommendation(row, matches_df, stats_df_with_cards)
        assert result is not None
        # 7 + 10 = 17 > 16.5 → WON
        assert result["won"] == True
        assert result["actual_value"] == 17

    def test_fouls_under(self, matches_df, stats_df_with_cards):
        from experiments.update_results import settle_recommendation

        row = pd.Series({
            "fixture_id": 100,
            "market": "FOULS",
            "side": "UNDER",
            "bet_type": "UNDER",
            "line": 16.5,
        })
        result = settle_recommendation(row, matches_df, stats_df_with_cards)
        assert result is not None
        # 17 < 16.5 is False → LOST
        assert result["won"] == False

    def test_corners_over(self, matches_df, stats_df_with_cards):
        from experiments.update_results import settle_recommendation

        row = pd.Series({
            "fixture_id": 100,
            "market": "CORNERS",
            "side": "OVER",
            "bet_type": "OVER",
            "line": 11.5,
        })
        result = settle_recommendation(row, matches_df, stats_df_with_cards)
        assert result is not None
        # 3 + 9 = 12 > 11.5 → WON
        assert result["won"] == True

    def test_home_win(self, matches_df, stats_df_with_cards):
        from experiments.update_results import settle_recommendation

        row = pd.Series({
            "fixture_id": 100,
            "market": "HOME_WIN",
            "side": "",
            "bet_type": "HOME_WIN",
            "line": 0,
        })
        result = settle_recommendation(row, matches_df, stats_df_with_cards)
        assert result is not None
        # 2 > 1 → WON
        assert result["won"] == True

    def test_strategy_scores_fouls(self):
        """Verify fouls still work in evaluate_market."""
        from experiments.update_strategy_scores import evaluate_market

        fixture = {"status": "FT", "home_goals": 1, "away_goals": 0}
        stats = {
            "home": {"fouls": 12, "yellow_cards": 2},
            "away": {"fouls": 9, "yellow_cards": 1},
        }
        result = evaluate_market("fouls", fixture, stats, line=20.5)
        assert result is not None
        won, actual = result
        # 12 + 9 = 21 > 20.5
        assert won == True

    def test_strategy_scores_corners(self):
        from experiments.update_strategy_scores import evaluate_market

        fixture = {"status": "FT", "home_goals": 2, "away_goals": 2}
        stats = {
            "home": {"corner_kicks": 5, "yellow_cards": 1},
            "away": {"corner_kicks": 4, "yellow_cards": 2},
        }
        result = evaluate_market("corners_over_85", fixture, stats, line=8.5)
        assert result is not None
        won, actual = result
        # 5 + 4 = 9 > 8.5
        assert won == True
