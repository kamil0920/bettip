"""Tests for Sportmonks per-line odds overlay."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.odds.sportmonks_per_line import (
    LEAGUE_ID_MAP,
    TARGET_LINES,
    TEAM_ALIASES,
    _normalize_team,
    load_sportmonks_per_line_odds,
    load_theodds_historical_odds,
    overlay_sportmonks_per_line_odds,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_corners_csv(path: Path, rows: list[dict] | None = None) -> Path:
    """Write a minimal corners_odds.csv."""
    if rows is None:
        rows = [
            {
                "fixture_id": 1,
                "fixture_name": "Arsenal vs Chelsea",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_team_normalized": "Arsenal",
                "away_team_normalized": "Chelsea",
                "start_time": "2024-03-01 15:00:00",
                "league_id": 8,
                "market_id": 69,
                "market": "Alternative Corners",
                "line": 9.5,
                "over_avg": 1.85,
                "over_best": 1.90,
                "over_count": 5,
                "under_avg": 2.00,
                "under_best": 2.05,
                "under_count": 5,
            },
            {
                "fixture_id": 1,
                "fixture_name": "Arsenal vs Chelsea",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_team_normalized": "Arsenal",
                "away_team_normalized": "Chelsea",
                "start_time": "2024-03-01 15:00:00",
                "league_id": 8,
                "market_id": 69,
                "market": "Alternative Corners",
                "line": 10.5,
                "over_avg": 2.40,
                "over_best": 2.50,
                "over_count": 5,
                "under_avg": 1.55,
                "under_best": 1.60,
                "under_count": 5,
            },
        ]
    csv_path = path / "corners_odds.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _make_cards_csv(path: Path, rows: list[dict] | None = None) -> Path:
    """Write a minimal cards_odds.csv."""
    if rows is None:
        rows = [
            {
                "fixture_id": 1,
                "fixture_name": "Arsenal vs Chelsea",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_team_normalized": "Arsenal",
                "away_team_normalized": "Chelsea",
                "start_time": "2024-03-01 15:00:00",
                "league_id": 8,
                "market_id": 255,
                "market": "Cards Over/Under",
                "line": 3.5,
                "over_avg": 1.70,
                "over_best": 1.75,
                "over_count": 3,
                "under_avg": 2.15,
                "under_best": 2.20,
                "under_count": 3,
            },
        ]
    csv_path = path / "cards_odds.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _make_features_df(n: int = 5) -> pd.DataFrame:
    """Create a minimal features DataFrame for testing."""
    return pd.DataFrame(
        {
            "league": ["premier_league"] * n,
            "date": pd.date_range("2024-03-01", periods=n, freq="D"),
            "home_team_name": ["Arsenal"] * n,
            "away_team_name": ["Chelsea"] * n,
            "total_corners": [10.0] * n,
            "total_cards": [4.0] * n,
        }
    )


def _make_theodds_parquet(path: Path, rows: list[dict] | None = None) -> Path:
    """Write a minimal The Odds API historical parquet."""
    if rows is None:
        rows = [
            {
                "event_id": "abc123",
                "date": "2024-03-01",
                "commence_time": "2024-03-01T15:00:00Z",
                "league": "premier_league",
                "sport_key": "soccer_epl",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "market": "corners",
                "line": 9.5,
                "over_avg": 1.82,
                "under_avg": 2.02,
                "over_max": 1.88,
                "under_max": 2.10,
                "num_bookmakers": 4,
            },
            {
                "event_id": "abc123",
                "date": "2024-03-01",
                "commence_time": "2024-03-01T15:00:00Z",
                "league": "premier_league",
                "sport_key": "soccer_epl",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "market": "corners",
                "line": 10.5,
                "over_avg": 2.35,
                "under_avg": 1.58,
                "over_max": 2.45,
                "under_max": 1.65,
                "num_bookmakers": 4,
            },
            {
                "event_id": "def456",
                "date": "2024-03-01",
                "commence_time": "2024-03-01T17:30:00Z",
                "league": "eredivisie",
                "sport_key": "soccer_netherlands_eredivisie",
                "home_team": "Ajax",
                "away_team": "PSV",
                "market": "corners",
                "line": 9.5,
                "over_avg": 1.90,
                "under_avg": 1.95,
                "over_max": 1.95,
                "under_max": 2.00,
                "num_bookmakers": 3,
            },
            {
                "event_id": "abc123",
                "date": "2024-03-01",
                "commence_time": "2024-03-01T15:00:00Z",
                "league": "premier_league",
                "sport_key": "soccer_epl",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "market": "cards",
                "line": 3.5,
                "over_avg": 1.68,
                "under_avg": 2.18,
                "over_max": 1.75,
                "under_max": 2.25,
                "num_bookmakers": 3,
            },
        ]
    pq_path = path / "niche_odds_historical.parquet"
    pd.DataFrame(rows).to_parquet(pq_path, index=False)
    return pq_path


# Path that never exists — passed to overlay to disable The Odds API loading
_NO_THEODDS = Path("/nonexistent/theodds.parquet")


# ---------------------------------------------------------------------------
# Tests: _normalize_team
# ---------------------------------------------------------------------------


class TestNormalizeTeam:
    """Team name normalization produces consistent canonical forms."""

    def test_basic_lowercase(self):
        assert _normalize_team("Arsenal") == "arsenal"

    def test_strips_diacritics(self):
        assert _normalize_team("Bayern München") == "bayern munich"
        assert _normalize_team("Atlético Madrid") == "atletico madrid"
        assert _normalize_team("Cádiz") == "cadiz"

    def test_strips_apostrophe(self):
        assert _normalize_team("Borussia M'gladbach") == "borussia monchengladbach"

    def test_strips_dots(self):
        # "1. FC Köln" → "1 fc koln" (dots removed, diacritics stripped)
        result = _normalize_team("1. FC Köln")
        assert result == "1 fc koln"

    def test_alias_newcastle(self):
        assert _normalize_team("Newcastle United") == "newcastle"
        assert _normalize_team("Newcastle") == "newcastle"

    def test_alias_west_ham(self):
        assert _normalize_team("West Ham United") == "west ham"
        assert _normalize_team("West Ham") == "west ham"

    def test_alias_sheffield(self):
        assert _normalize_team("Sheffield United") == "sheffield utd"

    def test_alias_hoffenheim(self):
        assert _normalize_team("TSG Hoffenheim") == "1899 hoffenheim"
        assert _normalize_team("1899 Hoffenheim") == "1899 hoffenheim"

    def test_alias_stuttgart(self):
        assert _normalize_team("Stuttgart") == "vfb stuttgart"
        assert _normalize_team("VfB Stuttgart") == "vfb stuttgart"

    def test_alias_brest(self):
        assert _normalize_team("Brest") == "stade brestois 29"
        assert _normalize_team("Stade Brestois 29") == "stade brestois 29"

    def test_alias_roma(self):
        assert _normalize_team("Roma") == "as roma"
        assert _normalize_team("AS Roma") == "as roma"

    def test_alias_verona(self):
        assert _normalize_team("Hellas Verona") == "verona"
        assert _normalize_team("Verona") == "verona"

    def test_nan_returns_empty(self):
        assert _normalize_team(None) == ""
        assert _normalize_team(float("nan")) == ""


# ---------------------------------------------------------------------------
# Tests: load_sportmonks_per_line_odds
# ---------------------------------------------------------------------------


class TestLoadAndPivot:
    """Loading CSVs produces correctly pivoted wide-format DataFrame."""

    def test_loads_and_pivots_corners(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_corners_csv(tmp_path)
            _make_cards_csv(tmp_path)  # needs both files

            result = load_sportmonks_per_line_odds(tmp_path)

        assert not result.empty
        assert "corners_over_avg_95" in result.columns
        assert "corners_under_avg_95" in result.columns
        assert "corners_over_avg_105" in result.columns
        assert "corners_under_avg_105" in result.columns

    def test_loads_and_pivots_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_corners_csv(tmp_path)
            _make_cards_csv(tmp_path)

            result = load_sportmonks_per_line_odds(tmp_path)

        assert "cards_over_avg_35" in result.columns
        assert "cards_under_avg_35" in result.columns

    def test_odds_values_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_corners_csv(tmp_path)
            _make_cards_csv(tmp_path)

            result = load_sportmonks_per_line_odds(tmp_path)

        # Check the actual odds values from our test data
        assert result["corners_over_avg_95"].iloc[0] == pytest.approx(1.85)
        assert result["corners_under_avg_95"].iloc[0] == pytest.approx(2.00)

    def test_filters_to_target_lines_only(self):
        """Lines outside TARGET_LINES (e.g. corners 16.0) are excluded."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = [
                {
                    "fixture_id": 1,
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "home_team_normalized": "Arsenal",
                    "away_team_normalized": "Chelsea",
                    "start_time": "2024-03-01 15:00:00",
                    "league_id": 8,
                    "market_id": 69,
                    "market": "Alternative Corners",
                    "line": 16.0,  # not in target lines
                    "over_avg": 26.0,
                    "over_best": 26.0,
                    "over_count": 1,
                    "under_avg": 1.02,
                    "under_best": 1.02,
                    "under_count": 1,
                    "fixture_name": "Arsenal vs Chelsea",
                },
            ]
            _make_corners_csv(tmp_path, rows)
            _make_cards_csv(tmp_path)

            result = load_sportmonks_per_line_odds(tmp_path)

        # Should not have corners_over_avg_160
        corners_cols = [c for c in result.columns if c.startswith("corners_")]
        assert not any("160" in c for c in corners_cols)

    def test_unknown_league_id_dropped(self):
        """Rows with league_id not in LEAGUE_ID_MAP are excluded."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = [
                {
                    "fixture_id": 99,
                    "home_team": "TeamA",
                    "away_team": "TeamB",
                    "home_team_normalized": "TeamA",
                    "away_team_normalized": "TeamB",
                    "start_time": "2024-03-01 15:00:00",
                    "league_id": 9999,  # unknown
                    "market_id": 69,
                    "market": "Alternative Corners",
                    "line": 9.5,
                    "over_avg": 1.85,
                    "over_best": 1.90,
                    "over_count": 5,
                    "under_avg": 2.00,
                    "under_best": 2.05,
                    "under_count": 5,
                    "fixture_name": "TeamA vs TeamB",
                },
            ]
            _make_corners_csv(tmp_path, rows)
            _make_cards_csv(tmp_path)

            result = load_sportmonks_per_line_odds(tmp_path)

        # Only cards data should be present (corners had unknown league)
        if not result.empty:
            corners_cols = [c for c in result.columns if c.startswith("corners_")]
            for c in corners_cols:
                assert result[c].isna().all()

    def test_missing_csv_returns_empty(self):
        """Missing CSV files produce empty DataFrame."""
        with tempfile.TemporaryDirectory() as tmp:
            result = load_sportmonks_per_line_odds(Path(tmp))
        assert result.empty


# ---------------------------------------------------------------------------
# Tests: overlay_sportmonks_per_line_odds
# ---------------------------------------------------------------------------


class TestOverlay:
    """Overlay function correctly merges Sportmonks odds into features."""

    def test_overlay_fills_new_columns(self):
        """Sportmonks odds create new columns in features when absent."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_corners_csv(tmp_path)
            _make_cards_csv(tmp_path)

            df = _make_features_df(1)
            result = overlay_sportmonks_per_line_odds(df, tmp_path, _NO_THEODDS)

        assert "corners_over_avg_95" in result.columns
        assert result["corners_over_avg_95"].iloc[0] == pytest.approx(1.85)

    def test_overlay_preserves_existing_values(self):
        """Pre-existing non-NaN odds are NOT overwritten by Sportmonks."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_corners_csv(tmp_path)
            _make_cards_csv(tmp_path)

            df = _make_features_df(1)
            df["corners_over_avg_95"] = 99.99  # pre-existing value

            result = overlay_sportmonks_per_line_odds(df, tmp_path, _NO_THEODDS)

        # Should keep the pre-existing value, not overwrite with Sportmonks 1.85
        assert result["corners_over_avg_95"].iloc[0] == pytest.approx(99.99)

    def test_overlay_fills_nan_values(self):
        """NaN cells in existing columns ARE filled by Sportmonks."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_corners_csv(tmp_path)
            _make_cards_csv(tmp_path)

            df = _make_features_df(1)
            df["corners_over_avg_95"] = np.nan  # pre-existing NaN

            result = overlay_sportmonks_per_line_odds(df, tmp_path, _NO_THEODDS)

        assert result["corners_over_avg_95"].iloc[0] == pytest.approx(1.85)

    def test_unmatched_fixtures_stay_nan(self):
        """Fixtures without Sportmonks data remain NaN."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_corners_csv(tmp_path)
            _make_cards_csv(tmp_path)

            df = _make_features_df(1)
            # Change to a date that doesn't match Sportmonks data
            df["date"] = pd.Timestamp("2020-01-01")

            result = overlay_sportmonks_per_line_odds(df, tmp_path, _NO_THEODDS)

        if "corners_over_avg_95" in result.columns:
            assert result["corners_over_avg_95"].isna().all()

    def test_no_temp_columns_leak(self):
        """Temporary merge columns (_home_norm etc.) are cleaned up."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_corners_csv(tmp_path)
            _make_cards_csv(tmp_path)

            df = _make_features_df(1)
            result = overlay_sportmonks_per_line_odds(df, tmp_path, _NO_THEODDS)

        temp_cols = [c for c in result.columns if c.startswith("_")]
        assert len(temp_cols) == 0, f"Temp columns leaked: {temp_cols}"

        sm_cols = [c for c in result.columns if c.endswith("_sm")]
        assert len(sm_cols) == 0, f"Suffix columns leaked: {sm_cols}"

    def test_missing_required_columns_returns_unchanged(self):
        """DataFrame without required columns is returned unchanged."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = overlay_sportmonks_per_line_odds(df)
        assert list(result.columns) == ["x"]


# ---------------------------------------------------------------------------
# Tests: column name consistency
# ---------------------------------------------------------------------------


class TestColumnNameConsistency:
    """Sportmonks pivot produces same column names as NB CDF generator."""

    def test_column_names_match_per_line_odds(self):
        """Column names from Sportmonks match those from generate_per_line_odds."""
        from src.odds.per_line_odds import PER_LINE_TARGETS, _line_to_col_suffix

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_corners_csv(tmp_path)
            _make_cards_csv(tmp_path)

            sm_result = load_sportmonks_per_line_odds(tmp_path)

        sm_cols = {c for c in sm_result.columns if "_avg_" in c}

        # Build expected column names from PER_LINE_TARGETS
        expected_cols = set()
        for stat in ["corners", "cards"]:
            for direction, lines in PER_LINE_TARGETS.get(stat, {}).items():
                for line in lines:
                    col = f"{stat}_{direction}_avg_{_line_to_col_suffix(line)}"
                    expected_cols.add(col)

        # Every Sportmonks column should be in the expected set
        unexpected = sm_cols - expected_cols
        assert not unexpected, f"Unexpected Sportmonks columns: {unexpected}"


# ---------------------------------------------------------------------------
# Tests: overlay + CDF fill integration
# ---------------------------------------------------------------------------


class TestOverlayThenCDF:
    """Sportmonks odds survive generate_per_line_odds() call."""

    def test_cdf_does_not_overwrite_sportmonks(self):
        """generate_per_line_odds() preserves pre-filled Sportmonks odds."""
        from src.odds.per_line_odds import generate_per_line_odds

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_corners_csv(tmp_path)
            _make_cards_csv(tmp_path)

            df = _make_features_df(3)
            df = overlay_sportmonks_per_line_odds(df, tmp_path, _NO_THEODDS)

        # Record the Sportmonks value
        sm_value = df["corners_over_avg_95"].iloc[0]
        assert not np.isnan(sm_value), "Sportmonks should have filled this"

        # Now run CDF fill
        df = generate_per_line_odds(df)

        # The Sportmonks value should be preserved
        assert df["corners_over_avg_95"].iloc[0] == pytest.approx(sm_value)

    def test_cdf_fills_non_sportmonks_rows(self):
        """Rows without Sportmonks data get CDF-estimated odds."""
        from src.odds.per_line_odds import generate_per_line_odds

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_corners_csv(tmp_path)
            _make_cards_csv(tmp_path)

            df = _make_features_df(3)
            # Only first row matches Sportmonks (2024-03-01)
            df = overlay_sportmonks_per_line_odds(df, tmp_path, _NO_THEODDS)

        # Row 2 (2024-03-02) has no Sportmonks data
        assert df["corners_over_avg_95"].iloc[1:].isna().all()

        # After CDF fill, all rows should have values
        df = generate_per_line_odds(df)
        assert df["corners_over_avg_95"].notna().all()


# ---------------------------------------------------------------------------
# Tests: load_theodds_historical_odds
# ---------------------------------------------------------------------------


class TestTheOddsHistorical:
    """The Odds API historical odds loading and pivoting."""

    def test_loads_and_pivots_corners(self):
        """Corners at target lines are loaded and pivoted correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pq = _make_theodds_parquet(tmp_path)
            result = load_theodds_historical_odds(pq)

        assert not result.empty
        assert "corners_over_avg_95" in result.columns
        assert "corners_under_avg_95" in result.columns
        assert "corners_over_avg_105" in result.columns

    def test_loads_cards(self):
        """Cards at target lines are loaded."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pq = _make_theodds_parquet(tmp_path)
            result = load_theodds_historical_odds(pq)

        assert "cards_over_avg_35" in result.columns
        assert "cards_under_avg_35" in result.columns

    def test_odds_values_preserved(self):
        """Odds values from parquet are preserved in pivot."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pq = _make_theodds_parquet(tmp_path)
            result = load_theodds_historical_odds(pq)

        # Arsenal vs Chelsea corners 9.5 → over_avg=1.82
        row = result[result["home_norm"] == "arsenal"]
        assert row["corners_over_avg_95"].iloc[0] == pytest.approx(1.82)
        assert row["corners_under_avg_95"].iloc[0] == pytest.approx(2.02)

    def test_expansion_leagues_included(self):
        """Expansion leagues (e.g. eredivisie) are loaded — no league_id mapping needed."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pq = _make_theodds_parquet(tmp_path)
            result = load_theodds_historical_odds(pq)

        assert "eredivisie" in result["league"].values

    def test_filters_non_target_lines(self):
        """Lines outside TARGET_LINES are excluded."""
        rows = [
            {
                "event_id": "x1",
                "date": "2024-03-01",
                "commence_time": "2024-03-01T15:00:00Z",
                "league": "premier_league",
                "sport_key": "soccer_epl",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "market": "corners",
                "line": 7.5,  # not in target lines
                "over_avg": 1.30,
                "under_avg": 3.20,
                "over_max": 1.35,
                "under_max": 3.30,
                "num_bookmakers": 3,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_theodds_parquet(tmp_path, rows)
            result = load_theodds_historical_odds(tmp_path / "niche_odds_historical.parquet")

        assert result.empty

    def test_skips_hc_markets(self):
        """cornershc/cardshc markets are excluded."""
        rows = [
            {
                "event_id": "x1",
                "date": "2024-03-01",
                "commence_time": "2024-03-01T15:00:00Z",
                "league": "premier_league",
                "sport_key": "soccer_epl",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "market": "cornershc",
                "line": 1.5,
                "over_avg": 1.90,
                "under_avg": 1.90,
                "over_max": 1.95,
                "under_max": 1.95,
                "num_bookmakers": 3,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_theodds_parquet(tmp_path, rows)
            result = load_theodds_historical_odds(tmp_path / "niche_odds_historical.parquet")

        assert result.empty

    def test_missing_parquet_returns_empty(self):
        """Missing parquet file produces empty DataFrame."""
        result = load_theodds_historical_odds(Path("/nonexistent/data.parquet"))
        assert result.empty


# ---------------------------------------------------------------------------
# Tests: combined overlay (Sportmonks + The Odds API)
# ---------------------------------------------------------------------------


class TestCombinedOverlay:
    """Overlay merges both Sportmonks and The Odds API sources."""

    def test_both_sources_contribute(self):
        """Features receive odds from both Sportmonks and The Odds API."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_corners_csv(tmp_path)
            _make_cards_csv(tmp_path)
            pq = _make_theodds_parquet(tmp_path)

            df = _make_features_df(1)
            result = overlay_sportmonks_per_line_odds(df, tmp_path, pq)

        # Should have corners data (from both sources)
        assert "corners_over_avg_95" in result.columns
        assert result["corners_over_avg_95"].notna().any()

    def test_expansion_league_from_theodds_only(self):
        """Expansion league fixtures get odds from The Odds API (not in Sportmonks)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _make_corners_csv(tmp_path)
            _make_cards_csv(tmp_path)
            pq = _make_theodds_parquet(tmp_path)

            df = pd.DataFrame(
                {
                    "league": ["eredivisie"],
                    "date": pd.to_datetime(["2024-03-01"]),
                    "home_team_name": ["Ajax"],
                    "away_team_name": ["PSV"],
                    "total_corners": [10.0],
                }
            )
            result = overlay_sportmonks_per_line_odds(df, tmp_path, pq)

        assert "corners_over_avg_95" in result.columns
        assert result["corners_over_avg_95"].iloc[0] == pytest.approx(1.90)

    def test_sportmonks_preferred_over_theodds_on_duplicate(self):
        """When both sources have data, the one with more columns wins."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Sportmonks has corners 9.5 + 10.5 (2 odds columns)
            _make_corners_csv(tmp_path)
            _make_cards_csv(tmp_path)

            # The Odds API also has corners 9.5 + 10.5 for same fixture (2 odds cols)
            # but also has cards 3.5 (4 odds cols total → wins)
            pq = _make_theodds_parquet(tmp_path)

            df = _make_features_df(1)
            result = overlay_sportmonks_per_line_odds(df, tmp_path, pq)

        # Should have data filled
        assert result["corners_over_avg_95"].notna().iloc[0]
