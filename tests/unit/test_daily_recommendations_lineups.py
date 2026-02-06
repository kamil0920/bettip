"""Tests for lineup integration in daily recommendations."""
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.generate_daily_recommendations import _load_prematch_lineups


class TestLoadPrematchLineups:
    """Test lineup loading from prematch cache."""

    def test_load_prematch_lineups_from_cache(self, tmp_path, monkeypatch):
        """Should load lineups from lineup_window_latest.json."""
        monkeypatch.setattr(
            "experiments.generate_daily_recommendations.project_root", tmp_path
        )

        # Create lineup file
        lineup_dir = tmp_path / "data" / "06-prematch" / "12345"
        lineup_dir.mkdir(parents=True)
        lineup_data = {
            "lineups": {
                "available": True,
                "home": {
                    "starting_xi": [
                        {"id": 101, "name": "Salah"},
                        {"id": 102, "name": "Nunez"},
                    ]
                },
                "away": {
                    "starting_xi": [
                        {"id": 201, "name": "Haaland"},
                    ]
                },
            }
        }
        with open(lineup_dir / "lineup_window_latest.json", "w") as f:
            json.dump(lineup_data, f)

        home, away = _load_prematch_lineups(12345)

        assert home is not None
        assert away is not None
        assert len(home["starting_xi"]) == 2
        assert home["starting_xi"][0]["id"] == 101

    def test_load_prematch_lineups_fixture_format(self, tmp_path, monkeypatch):
        """Should also check fixture_{id}.json format."""
        monkeypatch.setattr(
            "experiments.generate_daily_recommendations.project_root", tmp_path
        )

        prematch_dir = tmp_path / "data" / "06-prematch"
        prematch_dir.mkdir(parents=True)
        lineup_data = {
            "lineups": {
                "available": True,
                "home": {"starting_xi": [{"id": 101}]},
                "away": {"starting_xi": [{"id": 201}]},
            }
        }
        with open(prematch_dir / "fixture_99999.json", "w") as f:
            json.dump(lineup_data, f)

        home, away = _load_prematch_lineups(99999)

        assert home is not None
        assert away is not None

    def test_load_prematch_lineups_missing_returns_none(self, tmp_path, monkeypatch):
        """Missing lineup files should return (None, None)."""
        monkeypatch.setattr(
            "experiments.generate_daily_recommendations.project_root", tmp_path
        )

        # No lineup files exist
        prematch_dir = tmp_path / "data" / "06-prematch"
        prematch_dir.mkdir(parents=True)

        home, away = _load_prematch_lineups(12345)

        assert home is None
        assert away is None

    def test_load_prematch_lineups_not_available(self, tmp_path, monkeypatch):
        """Lineups with available=False should return (None, None)."""
        monkeypatch.setattr(
            "experiments.generate_daily_recommendations.project_root", tmp_path
        )

        lineup_dir = tmp_path / "data" / "06-prematch" / "12345"
        lineup_dir.mkdir(parents=True)
        lineup_data = {
            "lineups": {
                "available": False,
                "home": None,
                "away": None,
            }
        }
        with open(lineup_dir / "lineup_window_latest.json", "w") as f:
            json.dump(lineup_data, f)

        home, away = _load_prematch_lineups(12345)

        assert home is None
        assert away is None

    def test_load_prematch_lineups_malformed_json(self, tmp_path, monkeypatch):
        """Malformed JSON should return (None, None) gracefully."""
        monkeypatch.setattr(
            "experiments.generate_daily_recommendations.project_root", tmp_path
        )

        lineup_dir = tmp_path / "data" / "06-prematch" / "12345"
        lineup_dir.mkdir(parents=True)
        with open(lineup_dir / "lineup_window_latest.json", "w") as f:
            f.write("not json{}")

        home, away = _load_prematch_lineups(12345)

        assert home is None
        assert away is None
