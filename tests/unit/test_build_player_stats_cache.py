"""Tests for player stats cache builder."""
import numpy as np
import pandas as pd
import pytest

from scripts.build_player_stats_cache import build_player_stats_cache, _aggregate_across_seasons


def _create_player_stats_parquet(path, fixture_id, players):
    """Helper to create a player_stats.parquet file with given players."""
    rows = []
    for p in players:
        rows.append({
            "fixture_info": "{}",
            "players": "{}",
            "fixture_id": float(fixture_id),
            "date": "2024-01-01",
            "status": "FT",
            "home_team": "Team A",
            "away_team": "Team B",
            "score_home": 1.0,
            "score_away": 0.0,
            "team_name": p.get("team_name", "Team A"),
            "id": float(p["id"]),
            "name": p["name"],
            "photo": "",
            "offsides": 0.0,
            "games.minutes": float(p.get("minutes", 90)),
            "games.number": 10.0,
            "games.position": p.get("position", "M"),
            "games.rating": str(p.get("rating", "7.0")),
            "games.captain": "False",
            "games.substitute": "False",
            "shots.total": 0.0,
            "shots.on": 0.0,
            "goals.total": float(p.get("goals", 0)),
            "goals.conceded": 0.0,
            "goals.assists": float(p.get("assists", 0)),
            "goals.saves": 0.0,
            "passes.total": 50.0,
            "passes.key": 2.0,
            "passes.accuracy": "80",
            "tackles.total": 3.0,
            "tackles.blocks": 1.0,
            "tackles.interceptions": 2.0,
            "duels.total": 10.0,
            "duels.won": 5.0,
            "dribbles.attempts": 3.0,
            "dribbles.success": 1.0,
            "dribbles.past": 1.0,
            "fouls.drawn": 2.0,
            "fouls.committed": 1.0,
            "cards.yellow": 0.0,
            "cards.red": 0.0,
            "penalty.won": "None",
            "penalty.commited": "None",
            "penalty.scored": 0.0,
            "penalty.missed": 0.0,
            "penalty.saved": 0.0,
        })
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


class TestBuildPlayerStatsCache:
    """Test player stats cache building."""

    def test_aggregates_across_seasons(self, tmp_path):
        """Player appearing in multiple seasons should have weighted stats."""
        raw_dir = tmp_path / "raw"
        # Season 2023
        _create_player_stats_parquet(
            raw_dir / "premier_league" / "2023" / "player_stats.parquet",
            fixture_id=1001,
            players=[
                {"id": 101, "name": "Salah", "rating": "7.0", "goals": 1, "assists": 0, "minutes": 90, "position": "F"},
            ],
        )
        # Season 2024
        _create_player_stats_parquet(
            raw_dir / "premier_league" / "2024" / "player_stats.parquet",
            fixture_id=2001,
            players=[
                {"id": 101, "name": "Salah", "rating": "8.0", "goals": 2, "assists": 1, "minutes": 90, "position": "F"},
            ],
        )

        result = build_player_stats_cache(
            raw_data_dir=raw_dir,
            output_path=tmp_path / "cache" / "player_stats.parquet",
            min_matches=1,
        )

        assert len(result) == 1
        row = result.iloc[0]
        assert row["player_id"] == 101
        # 2024 weight=2.0, 2023 weight=1.5 → avg = (8.0*2 + 7.0*1.5)/(2+1.5) = 7.57
        assert row["avg_rating"] == pytest.approx(7.57, abs=0.1)

    def test_minimum_matches_filter(self, tmp_path):
        """Players below min_matches should be filtered out."""
        raw_dir = tmp_path / "raw"

        # Create multiple fixtures in one file for Regular (5 fixtures)
        # and only 1 for OneTimer
        rows = []
        for i in range(5):
            rows.append({
                "fixture_info": "{}",
                "players": "{}",
                "fixture_id": float(1001 + i),
                "date": f"2024-03-{10+i:02d}",
                "status": "FT",
                "home_team": "Team A",
                "away_team": "Team B",
                "score_home": 1.0,
                "score_away": 0.0,
                "team_name": "Team A",
                "id": 101.0,
                "name": "Regular",
                "photo": "",
                "offsides": 0.0,
                "games.minutes": 90.0,
                "games.number": 10.0,
                "games.position": "M",
                "games.rating": "7.0",
                "games.captain": "False",
                "games.substitute": "False",
                "shots.total": 0.0,
                "shots.on": 0.0,
                "goals.total": 0.0,
                "goals.conceded": 0.0,
                "goals.assists": 0.0,
                "goals.saves": 0.0,
                "passes.total": 50.0,
                "passes.key": 2.0,
                "passes.accuracy": "80",
                "tackles.total": 3.0,
                "tackles.blocks": 1.0,
                "tackles.interceptions": 2.0,
                "duels.total": 10.0,
                "duels.won": 5.0,
                "dribbles.attempts": 3.0,
                "dribbles.success": 1.0,
                "dribbles.past": 1.0,
                "fouls.drawn": 2.0,
                "fouls.committed": 1.0,
                "cards.yellow": 0.0,
                "cards.red": 0.0,
                "penalty.won": "None",
                "penalty.commited": "None",
                "penalty.scored": 0.0,
                "penalty.missed": 0.0,
                "penalty.saved": 0.0,
            })
        # OneTimer only in 1 fixture
        rows.append({
            **rows[0],
            "fixture_id": 1001.0,
            "id": 102.0,
            "name": "OneTimer",
        })
        df = pd.DataFrame(rows)
        path = raw_dir / "premier_league" / "2024" / "player_stats.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

        result = build_player_stats_cache(
            raw_data_dir=raw_dir,
            output_path=tmp_path / "cache" / "player_stats.parquet",
            min_matches=3,
        )

        # Regular has 5 matches, OneTimer has 1
        assert len(result) == 1
        assert result.iloc[0]["player_id"] == 101

    def test_goals_per_90_handles_zero_minutes(self, tmp_path):
        """Players with zero minutes should not cause division errors."""
        raw_dir = tmp_path / "raw"
        _create_player_stats_parquet(
            raw_dir / "premier_league" / "2024" / "player_stats.parquet",
            fixture_id=1001,
            players=[
                {"id": 101, "name": "Bench", "minutes": 0, "goals": 0},
                {"id": 102, "name": "Played", "minutes": 90, "goals": 2},
            ],
        )

        result = build_player_stats_cache(
            raw_data_dir=raw_dir,
            output_path=tmp_path / "cache" / "player_stats.parquet",
            min_matches=1,
        )

        # Bench player (0 minutes) should be excluded
        assert len(result) == 1
        assert result.iloc[0]["player_id"] == 102
        assert result.iloc[0]["goals_per_90"] == pytest.approx(2.0, rel=0.01)

    def test_output_schema_matches_injector(self, tmp_path):
        """Output schema must match what ExternalFeatureInjector expects."""
        raw_dir = tmp_path / "raw"
        _create_player_stats_parquet(
            raw_dir / "premier_league" / "2024" / "player_stats.parquet",
            fixture_id=1001,
            players=[
                {"id": 101, "name": "Player", "minutes": 90, "rating": "7.5", "position": "F"},
            ],
        )

        result = build_player_stats_cache(
            raw_data_dir=raw_dir,
            output_path=tmp_path / "cache" / "player_stats.parquet",
            min_matches=1,
        )

        required_cols = [
            "player_id", "player_name", "avg_rating",
            "total_minutes", "matches_played",
            "goals_per_90", "assists_per_90", "position",
        ]
        for col in required_cols:
            assert col in result.columns, f"Missing required column: {col}"

    def test_season_weighting(self, tmp_path):
        """2024 season should be weighted 2x vs 2022 season 1x."""
        raw_dir = tmp_path / "raw"
        # 2022 season: rating 6.0
        _create_player_stats_parquet(
            raw_dir / "premier_league" / "2022" / "player_stats.parquet",
            fixture_id=1001,
            players=[
                {"id": 101, "name": "Player", "rating": "6.0", "minutes": 90},
            ],
        )
        # 2024 season: rating 8.0
        _create_player_stats_parquet(
            raw_dir / "premier_league" / "2024" / "player_stats.parquet",
            fixture_id=2001,
            players=[
                {"id": 101, "name": "Player", "rating": "8.0", "minutes": 90},
            ],
        )

        result = build_player_stats_cache(
            raw_data_dir=raw_dir,
            output_path=tmp_path / "cache" / "player_stats.parquet",
            min_matches=1,
        )

        assert len(result) == 1
        # 2024 weight=2.0, 2022 weight=1.0 → avg = (8.0*2 + 6.0*1)/(2+1) = 7.33
        assert result.iloc[0]["avg_rating"] == pytest.approx(7.33, abs=0.1)

    def test_empty_raw_directory(self, tmp_path):
        """Empty raw directory should return empty DataFrame."""
        raw_dir = tmp_path / "empty_raw"
        raw_dir.mkdir()

        result = build_player_stats_cache(
            raw_data_dir=raw_dir,
            output_path=tmp_path / "cache" / "player_stats.parquet",
            min_matches=1,
        )

        assert result.empty
