"""Unit tests for LiveTracker."""

import json
import pytest
import numpy as np

from src.ml.live_tracker import LiveTracker


@pytest.fixture
def tracker(tmp_path):
    """Create a LiveTracker with temp directory."""
    return LiveTracker(
        output_dir=str(tmp_path / "live_tracking"),
        initial_bankroll=1000.0,
        kelly_fraction=0.25,
    )


class TestKellyStake:
    def test_positive_edge_returns_stake(self, tracker):
        # prob=0.55, odds=2.0 => edge exists
        stake = tracker._calculate_kelly_stake(0.55, 2.0)
        assert stake > 0

    def test_no_edge_returns_zero(self, tracker):
        # prob=0.45, odds=2.0 => no edge (expected value negative)
        stake = tracker._calculate_kelly_stake(0.45, 2.0)
        assert stake == 0

    def test_invalid_odds_returns_zero(self, tracker):
        assert tracker._calculate_kelly_stake(0.6, 1.0) == 0
        assert tracker._calculate_kelly_stake(0.6, 0.5) == 0

    def test_invalid_prob_returns_zero(self, tracker):
        assert tracker._calculate_kelly_stake(0.0, 2.0) == 0
        assert tracker._calculate_kelly_stake(1.0, 2.0) == 0

    def test_stake_capped_at_5pct(self, tracker):
        # Very high edge — should be capped
        stake = tracker._calculate_kelly_stake(0.95, 10.0)
        assert stake <= 0.05

    def test_fractional_kelly_applied(self, tracker):
        full_kelly_tracker = LiveTracker(
            output_dir=str(tracker.output_dir.parent / "full"),
            kelly_fraction=1.0,
        )
        # Use small edge so neither hits the 5% cap
        quarter_stake = tracker._calculate_kelly_stake(0.52, 2.0)
        full_stake = full_kelly_tracker._calculate_kelly_stake(0.52, 2.0)
        assert quarter_stake < full_stake
        assert quarter_stake == pytest.approx(full_stake * 0.25, rel=0.01)


class TestAddPrediction:
    def test_add_prediction_returns_record(self, tracker):
        pred = tracker.add_prediction(
            match_id="123",
            home_team="Liverpool",
            away_team="Man City",
            match_date="2026-03-07",
            league="premier_league",
            bet_type="home_win",
            our_probability=0.55,
            market_odds=2.0,
            threshold=0.50,
            meets_threshold=True,
            edge=0.05,
        )
        assert pred['home_team'] == "Liverpool"
        assert pred['away_team'] == "Man City"
        assert pred['our_probability'] == 0.55
        assert pred['status'] == 'pending'

    def test_prediction_saved_to_file(self, tracker):
        tracker.add_prediction(
            match_id="123",
            home_team="Liverpool",
            away_team="Man City",
            match_date="2026-03-07",
            league="premier_league",
            bet_type="home_win",
            our_probability=0.55,
            market_odds=2.0,
            threshold=0.50,
            meets_threshold=True,
        )
        assert tracker.predictions_file.exists()
        with open(tracker.predictions_file) as f:
            data = json.load(f)
        assert len(data) == 1

    def test_no_stake_when_threshold_not_met(self, tracker):
        pred = tracker.add_prediction(
            match_id="123",
            home_team="A",
            away_team="B",
            match_date="2026-03-07",
            league="test",
            bet_type="home_win",
            our_probability=0.45,
            market_odds=2.0,
            threshold=0.50,
            meets_threshold=False,
        )
        assert pred['recommended_stake'] == 0
        assert pred['recommended_amount'] == 0

    def test_implied_prob_calculated(self, tracker):
        pred = tracker.add_prediction(
            match_id="123",
            home_team="A",
            away_team="B",
            match_date="2026-03-07",
            league="test",
            bet_type="home_win",
            our_probability=0.55,
            market_odds=2.5,
            threshold=0.50,
            meets_threshold=True,
        )
        assert pred['implied_prob_at_prediction'] == pytest.approx(0.4, abs=1e-10)

    def test_key_generated_correctly(self, tracker):
        pred = tracker.add_prediction(
            match_id="123",
            home_team="Liverpool",
            away_team="Man City",
            match_date="2026-03-07T15:00:00",
            league="premier_league",
            bet_type="home_win",
            our_probability=0.55,
            market_odds=2.0,
            threshold=0.50,
            meets_threshold=True,
        )
        assert pred['key'] == "Liverpool_Man City_home_win_2026-03-07"


class TestRecordClosingOdds:
    def test_record_valid_closing_odds(self, tracker):
        tracker.add_prediction(
            match_id="123",
            home_team="A",
            away_team="B",
            match_date="2026-03-07",
            league="test",
            bet_type="home_win",
            our_probability=0.55,
            market_odds=2.0,
            threshold=0.50,
            meets_threshold=True,
        )
        key = "A_B_home_win_2026-03-07"
        result = tracker.record_closing_odds(key, 1.80)
        assert result is not None
        assert result['closing_odds'] == 1.80
        assert result['clv'] is not None
        assert result['status'] == 'has_closing'

    def test_record_unknown_key_returns_none(self, tracker):
        result = tracker.record_closing_odds("nonexistent_key", 1.80)
        assert result is None

    def test_record_invalid_odds_returns_none(self, tracker):
        tracker.add_prediction(
            match_id="123",
            home_team="A",
            away_team="B",
            match_date="2026-03-07",
            league="test",
            bet_type="home_win",
            our_probability=0.55,
            market_odds=2.0,
            threshold=0.50,
            meets_threshold=True,
        )
        key = "A_B_home_win_2026-03-07"
        result = tracker.record_closing_odds(key, 0.5)
        assert result is None


class TestStateLoadSave:
    def test_state_persists_across_instances(self, tmp_path):
        output_dir = str(tmp_path / "persist_test")
        t1 = LiveTracker(output_dir=output_dir)
        t1.add_prediction(
            match_id="1", home_team="A", away_team="B",
            match_date="2026-03-07", league="test",
            bet_type="home_win", our_probability=0.55,
            market_odds=2.0, threshold=0.50, meets_threshold=True,
        )
        # Create new instance from same directory
        t2 = LiveTracker(output_dir=output_dir)
        assert len(t2.predictions) == 1
        assert "A_B_home_win_2026-03-07" in t2.predictions
