"""Tests for bugfixes: referee features, SHAP string cleaning, duplicate EMA columns."""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. Referee feature engineer: "truth value of a Series is ambiguous" fix
# ---------------------------------------------------------------------------
class TestRefereeFeatureEngineer:
    """Test that RefereeFeatureEngineer handles all referee value types."""

    def _make_matches(self, referee_values):
        """Helper: build a minimal matches DataFrame with given referee values."""
        n = len(referee_values)
        return pd.DataFrame({
            'fixture_id': range(1, n + 1),
            'date': pd.date_range('2024-01-01', periods=n, freq='7D'),
            'home_team': ['Team A'] * n,
            'away_team': ['Team B'] * n,
            'referee': referee_values,
            'ft_home': [2] * n,
            'ft_away': [1] * n,
        })

    def test_string_referee(self):
        """Normal string referee should produce features without error."""
        from src.features.engineers.external import RefereeFeatureEngineer

        eng = RefereeFeatureEngineer(min_matches=1)
        matches = self._make_matches(['John Doe'] * 5)
        result = eng.create_features({'matches': matches})

        assert len(result) == 5
        assert 'ref_cards_avg' in result.columns

    def test_nan_referee(self):
        """NaN referee should fall back to defaults without error."""
        from src.features.engineers.external import RefereeFeatureEngineer

        eng = RefereeFeatureEngineer(min_matches=1)
        matches = self._make_matches([np.nan, np.nan, 'Jane Doe', np.nan, np.nan])
        result = eng.create_features({'matches': matches})

        assert len(result) == 5
        # NaN rows get default values
        assert result.iloc[0]['ref_matches'] == 0

    def test_none_referee(self):
        """None referee should fall back to defaults without error."""
        from src.features.engineers.external import RefereeFeatureEngineer

        eng = RefereeFeatureEngineer(min_matches=1)
        matches = self._make_matches([None, None, 'Ref X', None, None])
        result = eng.create_features({'matches': matches})

        assert len(result) == 5
        assert result.iloc[0]['ref_matches'] == 0

    def test_empty_string_referee(self):
        """Empty string referee should fall back to defaults."""
        from src.features.engineers.external import RefereeFeatureEngineer

        eng = RefereeFeatureEngineer(min_matches=1)
        matches = self._make_matches(['', '  ', 'Valid Ref', '', ''])
        result = eng.create_features({'matches': matches})

        assert len(result) == 5
        # Empty/whitespace-only should be treated as missing
        assert result.iloc[0]['ref_matches'] == 0

    def test_numeric_referee_not_crash(self):
        """Numeric value in referee column should not crash."""
        from src.features.engineers.external import RefereeFeatureEngineer

        eng = RefereeFeatureEngineer(min_matches=1)
        matches = self._make_matches([42, 3.14, 'Real Ref', 0, -1])
        result = eng.create_features({'matches': matches})

        assert len(result) == 5

    def test_mixed_types_no_series_error(self):
        """Mix of types (str, NaN, None, int) should never raise Series ambiguity."""
        from src.features.engineers.external import RefereeFeatureEngineer

        eng = RefereeFeatureEngineer(min_matches=1)
        matches = self._make_matches(['Ref A', np.nan, None, '', 'Ref B'])
        # This was the original bug - should not raise
        # "The truth value of a Series is ambiguous"
        result = eng.create_features({'matches': matches})
        assert len(result) == 5


# ---------------------------------------------------------------------------
# 2. SHAP bracketed string cleaning in load_data()
# ---------------------------------------------------------------------------
class TestBracketedStringCleaning:
    """Test that bracketed scientific notation strings are cleaned to floats."""

    def test_strip_brackets_from_scientific_notation(self):
        """Values like '[5.0743705E-1]' should become 0.507..."""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'home_team': ['A', 'B'],
            'away_team': ['B', 'A'],
            'feature_x': ['[5.0743705E-1]', '[3.9479554E-1]'],
            'feature_y': ['[1.234E0]', '[5.678E-2]'],
            'clean_feature': [0.5, 0.6],
        })

        # Apply the same cleaning logic from load_data()
        text_cols = {'date', 'home_team', 'away_team', 'league', 'season', 'referee'}
        for col in df.select_dtypes(include='object').columns:
            if col in text_cols:
                continue
            converted = pd.to_numeric(
                df[col].astype(str).str.strip('[]() '), errors='coerce'
            )
            if converted.notna().sum() >= df[col].notna().sum() * 0.5:
                df[col] = converted

        assert df['feature_x'].dtype in (np.float64, float)
        assert abs(df['feature_x'].iloc[0] - 0.5074) < 0.001
        assert abs(df['feature_y'].iloc[0] - 1.234) < 0.001

    def test_text_columns_not_converted(self):
        """Team names and other text columns should not be touched."""
        df = pd.DataFrame({
            'date': ['2024-01-01'],
            'home_team': ['Manchester United'],
            'away_team': ['Liverpool'],
            'feature_x': ['[0.5]'],
        })

        text_cols = {'date', 'home_team', 'away_team', 'league', 'season', 'referee'}
        for col in df.select_dtypes(include='object').columns:
            if col in text_cols:
                continue
            converted = pd.to_numeric(
                df[col].astype(str).str.strip('[]() '), errors='coerce'
            )
            if converted.notna().sum() >= df[col].notna().sum() * 0.5:
                df[col] = converted

        assert df['home_team'].iloc[0] == 'Manchester United'
        assert df['away_team'].iloc[0] == 'Liverpool'

    def test_mixed_string_and_numeric(self):
        """Columns with >50% convertible values should be cleaned."""
        df = pd.DataFrame({
            'feature': ['[0.5]', '[0.6]', 'not_a_number', '[0.7]'],
        })

        for col in df.select_dtypes(include='object').columns:
            converted = pd.to_numeric(
                df[col].astype(str).str.strip('[]() '), errors='coerce'
            )
            if converted.notna().sum() >= df[col].notna().sum() * 0.5:
                df[col] = converted

        # 3 out of 4 are convertible (75%), so column should be converted
        assert df['feature'].dtype in (np.float64, float)
        assert abs(df['feature'].iloc[0] - 0.5) < 0.001
        assert pd.isna(df['feature'].iloc[2])  # 'not_a_number' becomes NaN

    def test_mostly_text_column_not_converted(self):
        """Columns with <50% convertible values should stay as strings."""
        df = pd.DataFrame({
            'feature': ['hello', 'world', 'foo', '[0.5]'],
        })

        for col in df.select_dtypes(include='object').columns:
            converted = pd.to_numeric(
                df[col].astype(str).str.strip('[]() '), errors='coerce'
            )
            if converted.notna().sum() >= df[col].notna().sum() * 0.5:
                df[col] = converted

        # Only 1 out of 4 is convertible (25%), so column stays as object
        assert df['feature'].dtype == object
        assert df['feature'].iloc[0] == 'hello'


# ---------------------------------------------------------------------------
# 3. Duplicate EMA column rename: fouls_match_ema / shots_match_ema
# ---------------------------------------------------------------------------
class TestNicheMarketEMARename:
    """Test that niche market engineers use distinct column names."""

    def _make_fouls_data(self, n=20):
        """Helper: minimal DataFrame for FoulsFeatureEngineer."""
        teams = ['Team A', 'Team B', 'Team C', 'Team D']
        rows = []
        for i in range(n):
            home = teams[i % len(teams)]
            away = teams[(i + 1) % len(teams)]
            rows.append({
                'fixture_id': i + 1,
                'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=7 * i),
                'home_team': home,
                'away_team': away,
                'home_fouls': np.random.randint(8, 18),
                'away_fouls': np.random.randint(8, 18),
            })
        return pd.DataFrame(rows)

    def _make_shots_data(self, n=20):
        """Helper: minimal DataFrame for ShotsFeatureEngineer."""
        teams = ['Team A', 'Team B', 'Team C', 'Team D']
        rows = []
        for i in range(n):
            home = teams[i % len(teams)]
            away = teams[(i + 1) % len(teams)]
            rows.append({
                'fixture_id': i + 1,
                'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=7 * i),
                'home_team': home,
                'away_team': away,
                'home_shots': np.random.randint(5, 20),
                'away_shots': np.random.randint(5, 20),
                'home_shots_on_target': np.random.randint(2, 10),
                'away_shots_on_target': np.random.randint(2, 10),
            })
        return pd.DataFrame(rows)

    def test_fouls_creates_match_ema_not_committed_ema(self):
        """FoulsFeatureEngineer should create home_fouls_match_ema, not home_fouls_committed_ema."""
        from src.features.engineers.niche_markets import FoulsFeatureEngineer

        eng = FoulsFeatureEngineer(ema_span=5, min_matches=2)
        df = self._make_fouls_data()
        result = eng._build_features(df)

        # New names should exist
        assert 'home_fouls_match_ema' in result.columns
        assert 'away_fouls_match_ema' in result.columns

        # Old conflicting names should NOT exist
        assert 'home_fouls_committed_ema' not in result.columns
        assert 'away_fouls_committed_ema' not in result.columns

    def test_fouls_derived_features_use_match_ema(self):
        """Derived features (expected_fouls, fouls_diff) should work with renamed columns."""
        from src.features.engineers.niche_markets import FoulsFeatureEngineer

        eng = FoulsFeatureEngineer(ema_span=5, min_matches=2)
        df = self._make_fouls_data(30)
        result = eng._build_features(df)

        # Derived features should still be created
        assert 'expected_home_fouls' in result.columns
        assert 'expected_away_fouls' in result.columns
        assert 'expected_total_fouls' in result.columns
        assert 'fouls_diff' in result.columns

        # Derived features should have non-NaN values for later rows
        last_rows = result.tail(5)
        assert last_rows['expected_total_fouls'].notna().any()

    def test_shots_creates_match_ema_not_shots_ema(self):
        """ShotsFeatureEngineer should create home_shots_match_ema, not home_shots_ema."""
        from src.features.engineers.niche_markets import ShotsFeatureEngineer

        eng = ShotsFeatureEngineer(ema_span=5, min_matches=2)
        df = self._make_shots_data()
        result = eng._build_features(df)

        # New names should exist
        assert 'home_shots_match_ema' in result.columns
        assert 'away_shots_match_ema' in result.columns

        # Old conflicting names should NOT exist
        assert 'home_shots_ema' not in result.columns
        assert 'away_shots_ema' not in result.columns

    def test_shots_derived_features_use_match_ema(self):
        """Derived features (expected_shots, shots_attack_diff) should work with renamed columns."""
        from src.features.engineers.niche_markets import ShotsFeatureEngineer

        eng = ShotsFeatureEngineer(ema_span=5, min_matches=2)
        df = self._make_shots_data(30)
        result = eng._build_features(df)

        # Derived features should still be created
        assert 'expected_home_shots' in result.columns
        assert 'expected_away_shots' in result.columns
        assert 'expected_total_shots' in result.columns
        assert 'shots_attack_diff' in result.columns

        # Derived features should have non-NaN values for later rows
        last_rows = result.tail(5)
        assert last_rows['expected_total_shots'].notna().any()

    def test_shots_accuracy_uses_match_ema(self):
        """Shot accuracy calculation should use renamed match_ema columns."""
        from src.features.engineers.niche_markets import ShotsFeatureEngineer

        eng = ShotsFeatureEngineer(ema_span=5, min_matches=2)
        df = self._make_shots_data(30)
        result = eng._build_features(df)

        if 'home_shot_accuracy' in result.columns:
            # Should be computed without errors, values between 0 and ~2
            valid = result['home_shot_accuracy'].dropna()
            assert len(valid) > 0
            assert valid.min() >= 0


# ---------------------------------------------------------------------------
# 4. Column normalization: API-Football → pipeline standard names
# ---------------------------------------------------------------------------
class TestMatchStatsColumnNormalization:
    """Test that normalize_match_stats_columns renames API columns correctly."""

    def test_renames_all_known_api_columns(self):
        """All four API-Football naming variants should be renamed."""
        from src.data_collection.match_stats_utils import normalize_match_stats_columns

        df = pd.DataFrame({
            'fixture_id': [1],
            'home_corner_kicks': [5],
            'away_corner_kicks': [3],
            'home_total_shots': [12],
            'away_total_shots': [8],
            'home_shots_on_goal': [4],
            'away_shots_on_goal': [2],
            'home_ball_possession': [55],
            'away_ball_possession': [45],
        })

        result = normalize_match_stats_columns(df)

        assert 'home_corners' in result.columns
        assert 'away_corners' in result.columns
        assert 'home_shots' in result.columns
        assert 'away_shots' in result.columns
        assert 'home_shots_on_target' in result.columns
        assert 'away_shots_on_target' in result.columns
        assert 'home_possession' in result.columns
        assert 'away_possession' in result.columns

        # Old names should be gone
        assert 'home_corner_kicks' not in result.columns
        assert 'home_total_shots' not in result.columns
        assert 'home_shots_on_goal' not in result.columns
        assert 'home_ball_possession' not in result.columns

    def test_already_normalized_columns_unchanged(self):
        """Columns already using pipeline names should pass through unchanged."""
        from src.data_collection.match_stats_utils import normalize_match_stats_columns

        df = pd.DataFrame({
            'fixture_id': [1],
            'home_corners': [5],
            'away_corners': [3],
            'home_shots': [12],
            'away_shots': [8],
            'home_fouls': [10],
        })

        result = normalize_match_stats_columns(df)

        assert list(result.columns) == list(df.columns)
        assert result['home_corners'].iloc[0] == 5

    def test_values_preserved_after_rename(self):
        """Data values should be unchanged after renaming."""
        from src.data_collection.match_stats_utils import normalize_match_stats_columns

        df = pd.DataFrame({
            'fixture_id': [100, 200],
            'home_corner_kicks': [7, 4],
            'away_corner_kicks': [3, 6],
            'home_total_shots': [15, 10],
        })

        result = normalize_match_stats_columns(df)

        assert result['home_corners'].tolist() == [7, 4]
        assert result['away_corners'].tolist() == [3, 6]
        assert result['home_shots'].tolist() == [15, 10]

    def test_empty_dataframe(self):
        """Empty DataFrame should not raise."""
        from src.data_collection.match_stats_utils import normalize_match_stats_columns

        df = pd.DataFrame()
        result = normalize_match_stats_columns(df)
        assert result.empty

    def test_mixed_old_and_new_columns(self):
        """DataFrames with some old and some new names should handle both."""
        from src.data_collection.match_stats_utils import normalize_match_stats_columns

        df = pd.DataFrame({
            'fixture_id': [1],
            'home_corner_kicks': [5],   # old name
            'away_corners': [3],        # already correct
            'home_fouls': [10],         # unrelated, should stay
        })

        result = normalize_match_stats_columns(df)

        assert 'home_corners' in result.columns
        assert 'away_corners' in result.columns
        assert 'home_fouls' in result.columns
        assert result['home_corners'].iloc[0] == 5
        assert result['away_corners'].iloc[0] == 3

    def test_coalesces_duplicate_columns_after_concat(self):
        """After pd.concat of old + new seasons, old column should merge into new."""
        from src.data_collection.match_stats_utils import normalize_match_stats_columns

        # Simulate: old season has home_corner_kicks, new season has home_corners
        old_season = pd.DataFrame({
            'fixture_id': [1, 2],
            'home_corner_kicks': [5, 7],
            'home_corners': [np.nan, np.nan],  # absent in old data
        })
        new_season = pd.DataFrame({
            'fixture_id': [3, 4],
            'home_corner_kicks': [np.nan, np.nan],  # absent in new data
            'home_corners': [4, 6],
        })
        combined = pd.concat([old_season, new_season], ignore_index=True)

        result = normalize_match_stats_columns(combined)

        # Should have exactly ONE home_corners column, no duplicates
        assert list(result.columns).count('home_corners') == 1
        assert 'home_corner_kicks' not in result.columns
        # Values coalesced correctly
        assert result['home_corners'].tolist() == [5.0, 7.0, 4.0, 6.0]
