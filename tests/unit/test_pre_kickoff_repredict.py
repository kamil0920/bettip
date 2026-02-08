"""Tests for pre-kickoff re-prediction script."""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.pre_kickoff_repredict import (
    _find_morning_recs,
    _matches_in_window,
    compute_deltas,
    save_lineup_to_cache,
)


class TestPreKickoffRepredict:
    """Test pre-kickoff re-prediction components."""

    def test_detects_upgraded_predictions(self):
        """Predictions with increased probability should be flagged as UPGRADE."""
        morning_df = pd.DataFrame({
            "fixture_id": [123, 456],
            "market": ["SHOTS", "FOULS"],
            "bet_type": ["OVER", "OVER"],
            "probability": [0.65, 0.70],
            "home_team": ["Liverpool", "Arsenal"],
            "away_team": ["Chelsea", "Spurs"],
        })
        updated_df = pd.DataFrame({
            "fixture_id": [123, 456],
            "market": ["SHOTS", "FOULS"],
            "bet_type": ["OVER", "OVER"],
            "probability": [0.72, 0.68],  # SHOTS up, FOULS down
        })

        deltas = compute_deltas(morning_df, updated_df, delta_threshold=0.02)

        assert not deltas.empty
        shots_delta = deltas[deltas["market"] == "SHOTS"]
        assert len(shots_delta) == 1
        assert shots_delta.iloc[0]["action"] == "UPGRADE"
        assert shots_delta.iloc[0]["delta"] == pytest.approx(0.07, abs=0.01)

    def test_detects_downgraded_predictions(self):
        """Predictions with decreased probability should be flagged as DOWNGRADE."""
        morning_df = pd.DataFrame({
            "fixture_id": [123],
            "market": ["CORNERS"],
            "bet_type": ["OVER"],
            "probability": [0.75],
            "home_team": ["Man City"],
            "away_team": ["Brighton"],
        })
        updated_df = pd.DataFrame({
            "fixture_id": [123],
            "market": ["CORNERS"],
            "bet_type": ["OVER"],
            "probability": [0.68],  # Down by 7pp
        })

        deltas = compute_deltas(morning_df, updated_df, delta_threshold=0.02)

        assert len(deltas) == 1
        assert deltas.iloc[0]["action"] == "DOWNGRADE"
        assert deltas.iloc[0]["delta"] == pytest.approx(-0.07, abs=0.01)

    def test_preserves_morning_predictions_without_lineups(self):
        """Small deltas below threshold should be filtered out."""
        morning_df = pd.DataFrame({
            "fixture_id": [123],
            "market": ["SHOTS"],
            "bet_type": ["OVER"],
            "probability": [0.70],
        })
        updated_df = pd.DataFrame({
            "fixture_id": [123],
            "market": ["SHOTS"],
            "bet_type": ["OVER"],
            "probability": [0.705],  # Only 0.5pp change
        })

        deltas = compute_deltas(morning_df, updated_df, delta_threshold=0.02)

        # Below 2pp threshold — should be empty
        assert deltas.empty

    def test_budget_guard_limits_api_calls(self):
        """API budget should limit number of lineup fetches."""
        from scripts.pre_kickoff_repredict import fetch_lineups_from_api

        # Mock the API client at its source module
        with patch("src.data_collection.api_client.FootballAPIClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client._make_request.return_value = {
                "results": 2,
                "response": [
                    {"startXI": [{"player": {"id": 1, "name": "P1"}}], "formation": "4-3-3"},
                    {"startXI": [{"player": {"id": 2, "name": "P2"}}], "formation": "4-4-2"},
                ],
            }

            # Budget of 2 but 5 fixtures requested
            result = fetch_lineups_from_api([100, 200, 300, 400, 500], api_budget=2)

            # Should only fetch 2
            assert mock_client._make_request.call_count == 2

    def test_output_csv_schema(self, tmp_path):
        """Pre-kickoff CSV should have standard recommendation columns."""
        morning_df = pd.DataFrame({
            "fixture_id": [123],
            "date": ["2026-02-06"],
            "home_team": ["Liverpool"],
            "away_team": ["Chelsea"],
            "market": ["SHOTS"],
            "bet_type": ["OVER"],
            "probability": [0.70],
            "edge": [12.5],
        })
        updated_df = pd.DataFrame({
            "fixture_id": [123],
            "date": ["2026-02-06"],
            "home_team": ["Liverpool"],
            "away_team": ["Chelsea"],
            "market": ["SHOTS"],
            "bet_type": ["OVER"],
            "probability": [0.75],
            "edge": [15.2],
        })

        deltas = compute_deltas(morning_df, updated_df, delta_threshold=0.02)

        assert "delta" in deltas.columns
        assert "action" in deltas.columns
        assert "abs_delta" in deltas.columns

    def test_save_lineup_to_cache(self, tmp_path, monkeypatch):
        """Lineup data should be saved to prematch cache correctly."""
        monkeypatch.setattr(
            "scripts.pre_kickoff_repredict.project_root", tmp_path
        )

        lineup_data = {
            "home": {"starting_xi": [{"id": 101, "name": "Salah"}]},
            "away": {"starting_xi": [{"id": 201, "name": "Haaland"}]},
        }

        save_lineup_to_cache(12345, lineup_data)

        cache_file = tmp_path / "data" / "06-prematch" / "12345" / "lineup_window_latest.json"
        assert cache_file.exists()

        with open(cache_file) as f:
            saved = json.load(f)

        assert saved["lineups"]["available"] is True
        assert saved["lineups"]["home"]["starting_xi"][0]["id"] == 101

    def test_matches_in_window(self):
        """Should filter matches by kickoff window."""
        now = datetime.now(timezone.utc)
        matches = [
            {
                "fixture_id": 1,
                "home_team": "A",
                "away_team": "B",
                "kickoff": (now + timedelta(minutes=30)).isoformat(),
            },
            {
                "fixture_id": 2,
                "home_team": "C",
                "away_team": "D",
                "kickoff": (now + timedelta(minutes=120)).isoformat(),  # Too far
            },
            {
                "fixture_id": 3,
                "home_team": "E",
                "away_team": "F",
                "kickoff": (now + timedelta(minutes=5)).isoformat(),  # Too close
            },
        ]

        result = _matches_in_window(matches, window_start_min=15, window_end_min=70)

        assert len(result) == 1
        assert result[0]["fixture_id"] == 1

    def test_compute_deltas_empty_inputs(self):
        """Empty DataFrames should return empty result."""
        empty = pd.DataFrame()
        non_empty = pd.DataFrame({"fixture_id": [1], "market": ["X"], "bet_type": ["Y"], "probability": [0.5]})

        assert compute_deltas(empty, non_empty).empty
        assert compute_deltas(non_empty, empty).empty
        assert compute_deltas(empty, empty).empty
