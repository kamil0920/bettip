"""Tests that ALL_LEAGUES covers every league in LEAGUE_IDS."""
import pytest
from src.leagues import LEAGUE_IDS, ALL_LEAGUES, EUROPEAN_LEAGUES, AMERICAS_LEAGUES


class TestAllLeagues:
    def test_all_leagues_covers_every_league_id(self):
        """ALL_LEAGUES must include every key in LEAGUE_IDS."""
        assert set(ALL_LEAGUES) == set(LEAGUE_IDS.keys())

    def test_all_leagues_is_european_plus_americas(self):
        """ALL_LEAGUES = EUROPEAN_LEAGUES + AMERICAS_LEAGUES."""
        assert set(ALL_LEAGUES) == set(EUROPEAN_LEAGUES) | set(AMERICAS_LEAGUES)

    def test_no_duplicates_in_all_leagues(self):
        assert len(ALL_LEAGUES) == len(set(ALL_LEAGUES))


from pathlib import Path
import pandas as pd


class TestMatchStatsLoaderMixin:
    def test_mixin_loads_from_all_leagues(self, tmp_path):
        """Mixin must iterate ALL_LEAGUES, not just EUROPEAN_LEAGUES."""
        from src.features.engineers.base import MatchStatsLoaderMixin

        class TestEngineer(MatchStatsLoaderMixin):
            def __init__(self):
                self.data_dir = tmp_path

        # Create match_stats for mls (an AMERICAS league)
        mls_dir = tmp_path / "mls" / "2025"
        mls_dir.mkdir(parents=True)
        df = pd.DataFrame({
            'fixture_id': [1], 'home_corners': [5], 'away_corners': [3],
            'home_fouls': [10], 'away_fouls': [8], 'home_shots': [12],
            'away_shots': [9], 'date': ['2025-01-01'], 'home_team': ['A'],
            'away_team': ['B'],
        })
        df.to_parquet(mls_dir / "match_stats.parquet", index=False)

        eng = TestEngineer()
        result = eng._load_match_stats()
        assert len(result) == 1
        assert result.iloc[0]['league'] == 'mls'

    def test_mixin_returns_empty_df_when_no_data(self, tmp_path):
        from src.features.engineers.base import MatchStatsLoaderMixin

        class TestEngineer(MatchStatsLoaderMixin):
            def __init__(self):
                self.data_dir = tmp_path

        eng = TestEngineer()
        result = eng._load_match_stats()
        assert result.empty


class TestEngineersUseAllLeagues:
    """Verify all engineers that load match_stats use ALL_LEAGUES via mixin."""

    @pytest.mark.parametrize("engineer_module,engineer_class", [
        ("src.features.engineers.dynamics", "DynamicsFeatureEngineer"),
        ("src.features.engineers.niche_derived", "NicheStatDerivedFeatureEngineer"),
        ("src.features.engineers.entropy", "EntropyFeatureEngineer"),
        ("src.features.engineers.niche_markets", "FoulsFeatureEngineer"),
        ("src.features.engineers.niche_markets", "ShotsFeatureEngineer"),
        ("src.features.engineers.window_ratio", "WindowRatioFeatureEngineer"),
        ("src.features.engineers.offsides", "OffsidesFeatureEngineer"),
    ])
    def test_engineer_inherits_mixin(self, engineer_module, engineer_class):
        """Each engineer must inherit MatchStatsLoaderMixin."""
        import importlib
        from src.features.engineers.base import MatchStatsLoaderMixin
        mod = importlib.import_module(engineer_module)
        cls = getattr(mod, engineer_class)
        assert issubclass(cls, MatchStatsLoaderMixin), (
            f"{engineer_class} does not inherit MatchStatsLoaderMixin"
        )
