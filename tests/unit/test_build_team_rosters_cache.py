"""Tests for team rosters cache builder."""
import pandas as pd
import pytest

from scripts.build_team_rosters_cache import build_team_rosters_cache


def _create_lineups_parquet(path, fixtures):
    """
    Helper to create a lineups.parquet with given fixtures.

    fixtures: list of dicts, each with:
        fixture_id, date, home_team, away_team,
        starters: list of (team_name, player_id, player_name, pos)
    """
    rows = []
    for fix in fixtures:
        for team, pid, name, pos in fix["starters"]:
            rows.append({
                "fixture_info": "{}",
                "lineups": "{}",
                "fixture_id": float(fix["fixture_id"]),
                "date": fix["date"],
                "status": "FT",
                "home_team": fix["home_team"],
                "away_team": fix["away_team"],
                "score_home": 1.0,
                "score_away": 0.0,
                "team_name": team,
                "type": "StartXI",
                "id": float(pid),
                "name": name,
                "number": 10.0,
                "pos": pos,
                "grid": "1:1",
                "formation": "4-3-3",
            })
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _create_player_stats_cache(path, players):
    """Helper to create player stats cache parquet."""
    df = pd.DataFrame(players)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


class TestBuildTeamRostersCache:
    """Test team rosters cache building."""

    def test_expected_starters_from_lineup_history(self, tmp_path):
        """Players starting frequently should be identified as expected starters."""
        raw_dir = tmp_path / "raw"

        # 5 matches where Player A starts every time
        fixtures = []
        for i in range(5):
            fixtures.append({
                "fixture_id": 1000 + i,
                "date": f"2024-03-{10+i:02d}",
                "home_team": "Liverpool",
                "away_team": "Arsenal",
                "starters": [
                    ("Liverpool", 101, "Salah", "F"),
                    ("Liverpool", 102, "Nunez", "F"),
                    ("Arsenal", 201, "Saka", "F"),
                ],
            })

        _create_lineups_parquet(
            raw_dir / "premier_league" / "2024" / "lineups.parquet",
            fixtures,
        )

        # Player stats cache
        _create_player_stats_cache(
            tmp_path / "cache" / "player_stats.parquet",
            [
                {"player_id": 101, "player_name": "Salah", "avg_rating": 7.8, "position": "F"},
                {"player_id": 102, "player_name": "Nunez", "avg_rating": 7.2, "position": "F"},
                {"player_id": 201, "player_name": "Saka", "avg_rating": 7.5, "position": "F"},
            ],
        )

        result = build_team_rosters_cache(
            raw_data_dir=raw_dir,
            player_stats_cache_path=tmp_path / "cache" / "player_stats.parquet",
            output_path=tmp_path / "cache" / "team_rosters.parquet",
            lookback_matches=10,
            min_starts=3,
        )

        assert not result.empty
        # All 3 players started 5/5 matches, all should be expected starters
        liverpool_starters = result[result["team_name"] == "Liverpool"]
        assert len(liverpool_starters) == 2
        assert set(liverpool_starters["player_id"]) == {101, 102}

        # Ratings should come from player stats cache
        salah_row = result[result["player_id"] == 101].iloc[0]
        assert salah_row["avg_rating"] == pytest.approx(7.8, rel=0.01)

    def test_only_recent_matches_used(self, tmp_path):
        """Only the most recent N matches should be considered."""
        raw_dir = tmp_path / "raw"

        fixtures = []
        # Old matches (Player 101 starts)
        for i in range(5):
            fixtures.append({
                "fixture_id": 1000 + i,
                "date": f"2024-01-{10+i:02d}",
                "home_team": "Team A",
                "away_team": "Team B",
                "starters": [("Team A", 101, "OldStarter", "M")],
            })
        # Recent matches (Player 102 starts, Player 101 dropped)
        for i in range(5):
            fixtures.append({
                "fixture_id": 2000 + i,
                "date": f"2024-06-{10+i:02d}",
                "home_team": "Team A",
                "away_team": "Team B",
                "starters": [("Team A", 102, "NewStarter", "M")],
            })

        _create_lineups_parquet(
            raw_dir / "league" / "2024" / "lineups.parquet",
            fixtures,
        )

        result = build_team_rosters_cache(
            raw_data_dir=raw_dir,
            player_stats_cache_path=tmp_path / "nonexistent.parquet",
            output_path=tmp_path / "cache" / "team_rosters.parquet",
            lookback_matches=5,  # Only look at last 5 matches
            min_starts=3,
        )

        team_a = result[result["team_name"] == "Team A"]
        # Only NewStarter should be expected (5/5 in recent window)
        # OldStarter has 0/5 in recent window
        assert len(team_a) == 1
        assert team_a.iloc[0]["player_id"] == 102

    def test_minimum_starts_filter(self, tmp_path):
        """Players below min_starts threshold should be excluded."""
        raw_dir = tmp_path / "raw"

        fixtures = []
        for i in range(10):
            starters = [("Team A", 101, "Regular", "M")]
            if i < 2:  # Occasional player only starts 2 of 10
                starters.append(("Team A", 102, "Occasional", "M"))
            fixtures.append({
                "fixture_id": 1000 + i,
                "date": f"2024-03-{10+i:02d}",
                "home_team": "Team A",
                "away_team": "Team B",
                "starters": starters,
            })

        _create_lineups_parquet(
            raw_dir / "league" / "2024" / "lineups.parquet",
            fixtures,
        )

        result = build_team_rosters_cache(
            raw_data_dir=raw_dir,
            player_stats_cache_path=tmp_path / "nonexistent.parquet",
            output_path=tmp_path / "cache" / "team_rosters.parquet",
            lookback_matches=10,
            min_starts=3,
        )

        team_a = result[result["team_name"] == "Team A"]
        # Regular: 10 starts, Occasional: 2 starts (below min_starts=3)
        assert len(team_a) == 1
        assert team_a.iloc[0]["player_id"] == 101

    def test_empty_raw_directory(self, tmp_path):
        """Empty raw directory should return empty DataFrame."""
        raw_dir = tmp_path / "empty_raw"
        raw_dir.mkdir()

        result = build_team_rosters_cache(
            raw_data_dir=raw_dir,
            player_stats_cache_path=tmp_path / "nonexistent.parquet",
            output_path=tmp_path / "cache" / "team_rosters.parquet",
        )

        assert result.empty

    def test_output_schema(self, tmp_path):
        """Output should have required columns."""
        raw_dir = tmp_path / "raw"

        fixtures = [
            {
                "fixture_id": 1000 + i,
                "date": f"2024-03-{10+i:02d}",
                "home_team": "Team A",
                "away_team": "Team B",
                "starters": [("Team A", 101, "Player", "M")],
            }
            for i in range(5)
        ]
        _create_lineups_parquet(
            raw_dir / "league" / "2024" / "lineups.parquet",
            fixtures,
        )

        result = build_team_rosters_cache(
            raw_data_dir=raw_dir,
            player_stats_cache_path=tmp_path / "nonexistent.parquet",
            output_path=tmp_path / "cache" / "team_rosters.parquet",
            min_starts=3,
        )

        required_cols = [
            "team_name", "player_id", "player_name",
            "starts_in_last_n", "avg_rating", "position",
        ]
        for col in required_cols:
            assert col in result.columns, f"Missing required column: {col}"
