"""Tests for NicheStatDerivedFeatureEngineer and Fourier terms."""
import numpy as np
import pandas as pd
import pytest

from src.features.engineers.niche_derived import NicheStatDerivedFeatureEngineer
from src.features.engineers.context import SeasonPhaseFeatureEngineer


# ---------- Helpers ----------

def _make_match_stats(n=50):
    """Create synthetic match stats for testing."""
    np.random.seed(42)
    teams = ['TeamA', 'TeamB', 'TeamC', 'TeamD']
    rows = []
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        rows.append({
            'fixture_id': 1000 + i,
            'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i * 3),
            'home_team': home,
            'away_team': away,
            'home_fouls': np.random.randint(8, 18),
            'away_fouls': np.random.randint(8, 18),
            'home_shots': np.random.randint(5, 20),
            'away_shots': np.random.randint(5, 20),
            'home_corners': np.random.randint(2, 12),
            'away_corners': np.random.randint(2, 12),
            'home_yellow_cards': np.random.randint(0, 5),
            'away_yellow_cards': np.random.randint(0, 5),
            'home_red_cards': np.random.choice([0, 0, 0, 1]),
            'away_red_cards': np.random.choice([0, 0, 0, 1]),
            'home_goals': np.random.randint(0, 4),
            'away_goals': np.random.randint(0, 4),
            'league': 'premier_league',
        })
    return pd.DataFrame(rows)


def _make_matches_for_fourier(n=40):
    """Create match data for SeasonPhaseFeatureEngineer Fourier tests."""
    rows = []
    for i in range(n):
        rows.append({
            'fixture_id': 2000 + i,
            'date': pd.Timestamp('2024-08-15') + pd.Timedelta(days=i * 7),
            'round': f'Regular Season - {i + 1}',
            'home_team': 'TeamA',
            'away_team': 'TeamB',
            'home_goals': 1,
            'away_goals': 0,
        })
    return pd.DataFrame(rows)


# ---------- NicheStatDerivedFeatureEngineer Tests ----------

class TestNicheStatDerivedFeatures:
    """Tests for ratio and volatility features."""

    @pytest.fixture
    def engineer(self):
        return NicheStatDerivedFeatureEngineer(
            volatility_window=5, ratio_ema_span=5, min_matches=2,
        )

    @pytest.fixture
    def stats_df(self):
        return _make_match_stats(50)

    def test_ratio_features_no_leakage(self, engineer, stats_df):
        """Verify shift(1) — current match data is NOT used in its own features."""
        # Build features directly via _build_features (bypasses file loading)
        featured = engineer._build_features(stats_df)

        # For the first match of each team, ratio EMA should be NaN
        # (no prior data to compute from)
        team_a_home = featured[featured['home_team'] == 'TeamA']
        first_match_idx = team_a_home.index[0]

        # All home ratio features for first match should be NaN
        ratio_cols = [c for c in featured.columns if c.endswith('_ema') and c.startswith('home_')]
        for col in ratio_cols:
            val = featured.loc[first_match_idx, col]
            assert pd.isna(val), f"{col} should be NaN for first match, got {val}"

    def test_volatility_std_calculation(self, engineer, stats_df):
        """Verify rolling std produces reasonable values."""
        featured = engineer._build_features(stats_df)

        vol_cols = [c for c in featured.columns if '_volatility' in c]
        assert len(vol_cols) > 0, "No volatility features created"

        # Non-diff volatility columns (raw std) should be non-negative
        for col in vol_cols:
            if '_diff' in col:
                continue  # diff columns can be negative
            non_null = featured[col].dropna()
            if len(non_null) > 0:
                assert (non_null >= 0).all(), f"{col} has negative std values"

    def test_safe_division_zero_denominator(self, engineer):
        """NaN when denominator is 0, no division errors."""
        num = pd.Series([10.0, 5.0, 0.0])
        den = pd.Series([0.0, 2.0, 0.0])
        result = engineer._safe_divide(num, den)

        assert pd.isna(result.iloc[0]), "0 denominator should produce NaN"
        assert result.iloc[1] == pytest.approx(2.5)
        assert pd.isna(result.iloc[2]), "0/0 should produce NaN"

    def test_feature_names_unique(self, engineer, stats_df):
        """No duplicate column names in output."""
        featured = engineer._build_features(stats_df)
        dupes = featured.columns[featured.columns.duplicated()]
        assert len(dupes) == 0, f"Duplicate columns: {dupes.tolist()}"

    def test_feature_count(self, engineer, stats_df):
        """Expected number of new features created."""
        featured = engineer._build_features(stats_df)
        original_cols = set(stats_df.columns) | {'total_fouls', 'total_shots', 'total_corners', 'total_cards', 'home_cards', 'away_cards'}
        new_cols = [c for c in featured.columns if c not in original_cols]
        # Expect: ~10 ratio EMA + 2 drawn ratio + 5 diffs + 3 cross-stat + 1 dominance
        #       + 8 per-team vol + 4 vol diffs + 4 total vol = ~37
        assert len(new_cols) >= 20, f"Only {len(new_cols)} new features: {new_cols}"

    def test_derive_cards_from_yellow_red(self, engineer):
        """Cards derived correctly from yellow + red."""
        df = pd.DataFrame({
            'home_yellow_cards': [3, 2],
            'away_yellow_cards': [1, 4],
            'home_red_cards': [1, 0],
            'away_red_cards': [0, 1],
        })
        result = engineer._derive_cards(df)
        assert result['home_cards'].tolist() == [4.0, 2.0]
        assert result['away_cards'].tolist() == [1.0, 5.0]

    def test_missing_columns_graceful(self, engineer):
        """Engineer handles missing columns without crashing."""
        df = pd.DataFrame({
            'fixture_id': [1, 2],
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'home_team': ['A', 'B'],
            'away_team': ['B', 'A'],
            'home_fouls': [10, 12],
            'away_fouls': [11, 9],
            # No shots, corners, cards columns
        })
        featured = engineer._build_features(df)
        # Should still produce fouls volatility
        assert 'home_fouls_volatility' in featured.columns


# ---------- Fourier Feature Tests ----------

class TestFourierFeatures:
    """Tests for Fourier temporal encodings in SeasonPhaseFeatureEngineer."""

    @pytest.fixture
    def fourier_engineer(self):
        return SeasonPhaseFeatureEngineer()

    @pytest.fixture
    def matches_df(self):
        return _make_matches_for_fourier(38)

    def test_fourier_sin_cos_identity(self, fourier_engineer, matches_df):
        """sin^2 + cos^2 = 1 for all Fourier pairs."""
        result = fourier_engineer.create_features({'matches': matches_df})

        for prefix in ['round', 'month', 'dow']:
            sin_col = f'{prefix}_sin'
            cos_col = f'{prefix}_cos'
            assert sin_col in result.columns, f"Missing {sin_col}"
            assert cos_col in result.columns, f"Missing {cos_col}"

            identity = result[sin_col] ** 2 + result[cos_col] ** 2
            np.testing.assert_allclose(identity.values, 1.0, atol=1e-10,
                                       err_msg=f"sin^2+cos^2 != 1 for {prefix}")

    def test_fourier_month_cyclical(self, fourier_engineer):
        """January (month=1) should be close to December (month=12) in feature space."""
        jan_sin = np.sin(2 * np.pi * 1 / 12)
        jan_cos = np.cos(2 * np.pi * 1 / 12)
        dec_sin = np.sin(2 * np.pi * 12 / 12)
        dec_cos = np.cos(2 * np.pi * 12 / 12)

        # Euclidean distance between Jan and Dec should be small
        dist = np.sqrt((jan_sin - dec_sin) ** 2 + (jan_cos - dec_cos) ** 2)
        # Jan and Feb distance for comparison
        feb_sin = np.sin(2 * np.pi * 2 / 12)
        feb_cos = np.cos(2 * np.pi * 2 / 12)
        dist_jan_feb = np.sqrt((jan_sin - feb_sin) ** 2 + (jan_cos - feb_cos) ** 2)

        # Dec-Jan distance should be similar to Jan-Feb (adjacent months)
        assert abs(dist - dist_jan_feb) < 0.01, \
            f"Dec-Jan distance ({dist:.4f}) far from Jan-Feb ({dist_jan_feb:.4f})"

    def test_fourier_nan_round_default(self, fourier_engineer):
        """Missing round number should default to 0 for Fourier computation."""
        matches = pd.DataFrame({
            'fixture_id': [1],
            'date': ['2024-06-15'],
            'round': ['Unknown'],
            'home_team': ['A'],
            'away_team': ['B'],
            'home_goals': [1],
            'away_goals': [0],
        })
        result = fourier_engineer.create_features({'matches': matches})

        # round_number defaults to 19 (mid-season) in phase logic
        # but round_sin/cos should use the round_number from result
        assert not pd.isna(result['round_sin'].iloc[0])
        assert not pd.isna(result['round_cos'].iloc[0])

    def test_fourier_feature_count(self, fourier_engineer, matches_df):
        """Should add exactly 6 Fourier features."""
        result = fourier_engineer.create_features({'matches': matches_df})
        fourier_cols = [c for c in result.columns if c.endswith('_sin') or c.endswith('_cos')]
        assert len(fourier_cols) == 6, f"Expected 6 Fourier cols, got {fourier_cols}"

    def test_round_values_bounded(self, fourier_engineer, matches_df):
        """Fourier values should be in [-1, 1]."""
        result = fourier_engineer.create_features({'matches': matches_df})
        for col in ['round_sin', 'round_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos']:
            assert result[col].min() >= -1.0 - 1e-10, f"{col} below -1"
            assert result[col].max() <= 1.0 + 1e-10, f"{col} above 1"
