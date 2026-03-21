"""Tests for booking points computation utility."""

import numpy as np
import pandas as pd
import pytest

from src.utils.booking_points import (
    compute_booking_points_from_events,
    compute_booking_points_from_stats,
)


class TestComputeBookingPointsFromStats:
    """Stats-based booking points (yellow=1, red=2)."""

    def test_yellows_only(self):
        assert compute_booking_points_from_stats(3, 0) == 3

    def test_straight_red(self):
        """Straight red = 2 booking points."""
        assert compute_booking_points_from_stats(0, 1) == 2

    def test_yellows_and_reds(self):
        """4 yellows + 1 red = 4 + 2 = 6."""
        assert compute_booking_points_from_stats(4, 1) == 6

    def test_zero_cards(self):
        assert compute_booking_points_from_stats(0, 0) == 0

    def test_multiple_reds(self):
        """2 yellows + 2 reds = 2 + 4 = 6."""
        assert compute_booking_points_from_stats(2, 2) == 6

    def test_pandas_series(self):
        yellows = pd.Series([3, 0, 4])
        reds = pd.Series([0, 1, 1])
        result = compute_booking_points_from_stats(yellows, reds)
        expected = pd.Series([3, 2, 6])
        pd.testing.assert_series_equal(result, expected)

    def test_numpy_array(self):
        yellows = np.array([3, 0, 4])
        reds = np.array([0, 1, 1])
        result = compute_booking_points_from_stats(yellows, reds)
        np.testing.assert_array_equal(result, np.array([3, 2, 6]))

    def test_nan_propagation(self):
        """NaN in input should propagate to output."""
        yellows = pd.Series([3, np.nan])
        reds = pd.Series([1, 0])
        result = compute_booking_points_from_stats(yellows, reds)
        assert result.iloc[0] == 5
        assert pd.isna(result.iloc[1])


class TestComputeBookingPointsFromEvents:
    """Events-based exact booking points with 2Y->R detection."""

    def _make_events(self, rows):
        """Helper to create card events DataFrame.

        rows: list of (fixture_id, player_id, detail, is_home)
        """
        return pd.DataFrame(rows, columns=["fixture_id", "player_id", "detail", "is_home"])

    def test_yellows_only(self):
        """3 yellow cards, no reds."""
        events = self._make_events([
            (1, 101, "Yellow Card", True),
            (1, 102, "Yellow Card", True),
            (1, 201, "Yellow Card", False),
        ])
        result = compute_booking_points_from_events(events)
        row = result[result["fixture_id"] == 1].iloc[0]
        assert row["home_cards"] == 2
        assert row["away_cards"] == 1
        assert row["total_cards"] == 3

    def test_straight_red(self):
        """1 straight red = 2 booking points."""
        events = self._make_events([
            (1, 101, "Red Card", True),
        ])
        result = compute_booking_points_from_events(events)
        row = result[result["fixture_id"] == 1].iloc[0]
        assert row["home_cards"] == 2
        assert row["away_cards"] == 0
        assert row["total_cards"] == 2

    def test_second_yellow_to_red(self):
        """2Y->R: API sends Yellow, Yellow, Red for same player.

        Bookmaker convention: 1st yellow (1pt) + red (2pt) = 3pt.
        The 2nd yellow is absorbed (not counted separately).
        """
        events = self._make_events([
            (1, 101, "Yellow Card", True),   # 1st yellow
            (1, 101, "Yellow Card", True),   # 2nd yellow (absorbed)
            (1, 101, "Red Card", True),      # Red card = 2pt
            (1, 201, "Yellow Card", False),  # Another player's yellow
        ])
        result = compute_booking_points_from_events(events)
        row = result[result["fixture_id"] == 1].iloc[0]
        # Player 101: 2 yellows + 1 red - 1 absorbed = 3 booking pts
        assert row["home_cards"] == 3
        # Player 201: 1 yellow = 1 booking pt
        assert row["away_cards"] == 1
        assert row["total_cards"] == 4

    def test_mixed_straight_red_and_2y_r(self):
        """One straight red + one 2Y->R in same match."""
        events = self._make_events([
            # Player 101: straight red
            (1, 101, "Red Card", True),
            # Player 201: 2Y->R
            (1, 201, "Yellow Card", False),
            (1, 201, "Yellow Card", False),
            (1, 201, "Red Card", False),
            # Player 202: simple yellow
            (1, 202, "Yellow Card", False),
        ])
        result = compute_booking_points_from_events(events)
        row = result[result["fixture_id"] == 1].iloc[0]
        # Home: player 101 straight red = 2pt
        assert row["home_cards"] == 2
        # Away: player 201 (2Y->R) = 3pt + player 202 (yellow) = 1pt = 4pt
        assert row["away_cards"] == 4
        assert row["total_cards"] == 6

    def test_multiple_fixtures(self):
        """Multiple fixtures computed correctly."""
        events = self._make_events([
            (1, 101, "Yellow Card", True),
            (1, 201, "Yellow Card", False),
            (2, 301, "Red Card", True),
            (2, 401, "Yellow Card", False),
        ])
        result = compute_booking_points_from_events(events)
        assert len(result) == 2
        r1 = result[result["fixture_id"] == 1].iloc[0]
        r2 = result[result["fixture_id"] == 2].iloc[0]
        assert r1["total_cards"] == 2  # 1 + 1
        assert r2["total_cards"] == 3  # 2 (red) + 1 (yellow)

    def test_empty_events(self):
        """Empty events returns empty DataFrame with correct columns."""
        events = pd.DataFrame(columns=["fixture_id", "player_id", "detail", "is_home"])
        result = compute_booking_points_from_events(events)
        assert list(result.columns) == ["fixture_id", "total_cards", "home_cards", "away_cards"]
        assert len(result) == 0

    def test_only_home_cards(self):
        """Match where only home team got cards."""
        events = self._make_events([
            (1, 101, "Yellow Card", True),
            (1, 102, "Yellow Card", True),
        ])
        result = compute_booking_points_from_events(events)
        row = result[result["fixture_id"] == 1].iloc[0]
        assert row["home_cards"] == 2
        assert row["away_cards"] == 0
        assert row["total_cards"] == 2
