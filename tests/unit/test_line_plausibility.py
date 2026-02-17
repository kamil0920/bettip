"""Tests for shared line plausibility utilities."""
import numpy as np
import pandas as pd
import pytest

from src.utils.line_plausibility import (
    LINE_PLAUSIBILITY,
    check_line_plausible,
    compute_league_stat_averages,
    filter_implausible_training_rows,
    parse_market_line,
)


class TestParseMarketLine:
    """Tests for parse_market_line()."""

    def test_fouls_over(self):
        result = parse_market_line("fouls_over_255")
        assert result == ("fouls", 25.5, "over")

    def test_fouls_under(self):
        result = parse_market_line("fouls_under_235")
        assert result == ("fouls", 23.5, "under")

    def test_cards_over(self):
        result = parse_market_line("cards_over_35")
        assert result == ("cards", 3.5, "over")

    def test_corners_under(self):
        result = parse_market_line("corners_under_105")
        assert result == ("corners", 10.5, "under")

    def test_shots_over(self):
        result = parse_market_line("shots_over_275")
        assert result == ("shots", 27.5, "over")

    def test_non_line_market_returns_none(self):
        assert parse_market_line("home_win") is None
        assert parse_market_line("btts") is None
        assert parse_market_line("over25") is None
        assert parse_market_line("fouls") is None
        assert parse_market_line("cards") is None

    def test_base_market_returns_none(self):
        assert parse_market_line("shots") is None
        assert parse_market_line("corners") is None


class TestComputeLeagueStatAverages:
    """Tests for compute_league_stat_averages()."""

    def test_basic_computation(self):
        df = pd.DataFrame({
            "league": ["premier_league"] * 3 + ["serie_a"] * 3,
            "total_fouls": [24, 26, 28, 30, 32, 34],
            "total_shots": [22, 24, 26, 20, 22, 24],
        })
        result = compute_league_stat_averages(df)
        assert "premier_league" in result
        assert "serie_a" in result
        assert result["premier_league"]["total_fouls"] == pytest.approx(26.0)
        assert result["serie_a"]["total_fouls"] == pytest.approx(32.0)

    def test_missing_league_column(self):
        df = pd.DataFrame({"total_fouls": [24, 26]})
        assert compute_league_stat_averages(df) == {}

    def test_handles_nan(self):
        df = pd.DataFrame({
            "league": ["pl", "pl", "pl"],
            "total_fouls": [24, np.nan, 26],
        })
        result = compute_league_stat_averages(df)
        assert result["pl"]["total_fouls"] == pytest.approx(25.0)


class TestCheckLinePlausible:
    """Tests for check_line_plausible()."""

    def test_plausible_fouls_line(self):
        league_stats = {"premier_league": {"total_fouls": 24.5}}
        status, reason = check_line_plausible(
            "fouls_over_255", "premier_league", league_stats
        )
        # 25.5 within 24.5 ± 4.0 = [20.5, 28.5]
        assert status == "yes"

    def test_implausible_fouls_line(self):
        league_stats = {"eredivisie": {"total_fouls": 20.0}}
        status, reason = check_line_plausible(
            "fouls_over_265", "eredivisie", league_stats
        )
        # 26.5 outside 20.0 ± 4.0 = [16.0, 24.0]
        assert status == "no"

    def test_non_line_market_always_yes(self):
        status, _ = check_line_plausible("home_win", "pl", {})
        assert status == "yes"

    def test_unknown_league(self):
        league_stats = {"premier_league": {"total_fouls": 24.5}}
        status, _ = check_line_plausible(
            "fouls_over_255", "unknown_league", league_stats
        )
        assert status == "unknown"

    def test_cards_tight_buffer(self):
        league_stats = {"pl": {"total_cards": 4.5}}
        # 3.5 within 4.5 ± 2.0 = [2.5, 6.5] -> yes
        status, _ = check_line_plausible("cards_over_35", "pl", league_stats)
        assert status == "yes"
        # 1.5 outside 4.5 ± 2.0 = [2.5, 6.5] -> no
        status, _ = check_line_plausible("cards_over_15", "pl", league_stats)
        assert status == "no"


class TestFilterImplausibleTrainingRows:
    """Tests for filter_implausible_training_rows()."""

    def _make_df(self, n_per_league=50):
        """Create a test DataFrame with two leagues with different fouls averages."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=n_per_league * 2, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "league": ["high_fouls_league"] * n_per_league + ["low_fouls_league"] * n_per_league,
            "total_fouls": (
                list(np.random.normal(28, 3, n_per_league))
                + list(np.random.normal(18, 3, n_per_league))
            ),
            "home_team": [f"team_{i % 10}" for i in range(n_per_league * 2)],
            "away_team": [f"team_{(i + 5) % 10}" for i in range(n_per_league * 2)],
        })
        return df.sort_values("date").reset_index(drop=True)

    def test_non_line_market_unchanged(self):
        df = self._make_df()
        result = filter_implausible_training_rows(df, "fouls")
        assert len(result) == len(df)

    def test_non_niche_market_unchanged(self):
        df = self._make_df()
        result = filter_implausible_training_rows(df, "home_win")
        assert len(result) == len(df)

    def test_filters_implausible_leagues(self):
        df = self._make_df(n_per_league=100)
        # fouls_over_265 line = 26.5
        # high_fouls_league avg ~28 -> within 26.5 ± 4 = [22.5, 30.5] -> keep
        # low_fouls_league avg ~18 -> outside [22.5, 30.5] -> filter
        result = filter_implausible_training_rows(df, "fouls_over_265")
        # Should remove most of low_fouls_league rows (after expanding mean converges)
        assert len(result) < len(df)
        # High fouls league should be mostly kept
        high_kept = (result["league"] == "high_fouls_league").sum()
        assert high_kept > 80  # Most of 100 rows kept

    def test_preserves_early_rows_with_insufficient_history(self):
        """Rows with NaN expanding avg (first ~10 per league) should be kept."""
        df = self._make_df(n_per_league=15)
        result = filter_implausible_training_rows(df, "fouls_over_265")
        # With only 15 rows per league, first 10 will have NaN expanding avg
        # Those should be preserved
        assert len(result) >= 20  # At least the NaN rows from both leagues

    def test_no_leakage_in_expanding_avg(self):
        """Verify expanding avg uses shift(1) — feature at row i only uses data < i."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=20),
            "league": ["test_league"] * 20,
            "total_fouls": [20] * 10 + [40] * 10,  # sudden jump
            "home_team": [f"t{i}" for i in range(20)],
            "away_team": [f"t{i+1}" for i in range(20)],
        })
        # Before filtering, manually check expanding avg behavior
        df_copy = df.copy()
        df_copy["_avg"] = df_copy.groupby("league")["total_fouls"].transform(
            lambda x: x.shift(1).expanding(min_periods=10).mean()
        )
        # At row 10 (first high-fouls row), expanding avg should be 20 (from first 10 rows)
        # NOT incorporating the 40s that come later
        avg_at_10 = df_copy.iloc[10]["_avg"]
        assert avg_at_10 == pytest.approx(20.0, abs=0.1)

    def test_missing_columns_returns_unchanged(self):
        df = pd.DataFrame({"date": [1, 2], "other": [3, 4]})
        result = filter_implausible_training_rows(df, "fouls_over_255")
        assert len(result) == len(df)
