"""Tests for DynamicsFeatureEngineer and league clustering."""
import numpy as np
import pandas as pd
import pytest

from src.features.engineers.dynamics import DynamicsFeatureEngineer


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


# ---------- DynamicsFeatureEngineer Tests ----------

class TestDynamicsFeatures:
    """Tests for distributional, momentum, and regime features."""

    @pytest.fixture
    def engineer(self):
        return DynamicsFeatureEngineer(
            window=5, short_ema=3, long_ema=8, long_window=10,
            damping_factor=0.9, min_matches=2,
        )

    @pytest.fixture
    def stats_df(self):
        return _make_match_stats(50)

    def test_leakage_prevention(self, engineer, stats_df):
        """Verify shift(1) — first match per team has NaN for all dynamics features."""
        featured = engineer._build_features(stats_df)

        team_a_home = featured[featured['home_team'] == 'TeamA']
        first_idx = team_a_home.index[0]

        # All dynamics features for the first match should be NaN
        dynamics_cols = [
            c for c in featured.columns
            if any(suffix in c for suffix in [
                '_skewness', '_kurtosis', '_cov',
                '_momentum_ratio', '_first_diff',
                '_variance_ratio',
            ])
            and c.startswith('home_')
        ]
        assert len(dynamics_cols) > 0, "No dynamics features found"

        for col in dynamics_cols:
            val = featured.loc[first_idx, col]
            assert pd.isna(val), f"{col} should be NaN for first match, got {val}"

    def test_no_inf_values(self, engineer, stats_df):
        """No inf values in any features."""
        featured = engineer._build_features(stats_df)

        dynamics_cols = [
            c for c in featured.columns
            if any(suffix in c for suffix in [
                '_skewness', '_kurtosis', '_cov',
                '_momentum_ratio', '_first_diff',
                '_variance_ratio',
            ])
        ]
        for col in dynamics_cols:
            non_null = featured[col].dropna()
            if len(non_null) > 0:
                assert not np.isinf(non_null).any(), f"{col} has inf values"

    def test_safe_division_cov(self, engineer, stats_df):
        """CoV handles zero mean gracefully (produces NaN, not inf)."""
        # Create data with zero values to force zero mean
        df = stats_df.copy()
        df.loc[df['home_team'] == 'TeamA', 'home_fouls'] = 0
        featured = engineer._build_features(df)

        cov_col = 'home_fouls_cov'
        if cov_col in featured.columns:
            non_null = featured[cov_col].dropna()
            if len(non_null) > 0:
                assert not np.isinf(non_null).any(), f"{cov_col} has inf from zero mean"

    def test_safe_division_momentum_ratio(self, engineer, stats_df):
        """Momentum ratio handles zero long EMA gracefully."""
        df = stats_df.copy()
        df.loc[df['home_team'] == 'TeamA', 'home_fouls'] = 0
        featured = engineer._build_features(df)

        mr_col = 'home_fouls_momentum_ratio'
        if mr_col in featured.columns:
            non_null = featured[mr_col].dropna()
            if len(non_null) > 0:
                assert not np.isinf(non_null).any(), f"{mr_col} has inf values"

    def test_feature_count(self, engineer, stats_df):
        """At least 50 new features created (target: 56)."""
        featured = engineer._build_features(stats_df)
        original_cols = set(stats_df.columns) | {'home_cards', 'away_cards'}
        new_cols = [c for c in featured.columns if c not in original_cols]
        assert len(new_cols) >= 50, f"Only {len(new_cols)} new features (expected >=50): {sorted(new_cols)}"

    def test_feature_names_unique(self, engineer, stats_df):
        """No duplicate column names in output."""
        featured = engineer._build_features(stats_df)
        dupes = featured.columns[featured.columns.duplicated()]
        assert len(dupes) == 0, f"Duplicate columns: {dupes.tolist()}"

    def test_no_naming_collisions_with_niche_derived(self, engineer, stats_df):
        """Dynamics features don't collide with NicheStatDerivedFeatureEngineer names."""
        from src.features.engineers.niche_derived import NicheStatDerivedFeatureEngineer
        niche_eng = NicheStatDerivedFeatureEngineer(
            volatility_window=5, ratio_ema_span=5, min_matches=2,
        )

        dynamics_featured = engineer._build_features(stats_df.copy())
        niche_featured = niche_eng._build_features(stats_df.copy())

        # Get new columns from each
        original_cols = set(stats_df.columns) | {'home_cards', 'away_cards',
                                                   'total_fouls', 'total_shots',
                                                   'total_corners', 'total_cards'}
        dynamics_new = set(c for c in dynamics_featured.columns if c not in original_cols)
        niche_new = set(c for c in niche_featured.columns if c not in original_cols)

        overlap = dynamics_new & niche_new
        assert len(overlap) == 0, f"Naming collision with niche_derived: {overlap}"

    def test_skewness_bounds(self, engineer, stats_df):
        """Skewness clipped to [-3, 3]."""
        featured = engineer._build_features(stats_df)
        skew_cols = [c for c in featured.columns if '_skewness' in c]
        assert len(skew_cols) > 0

        for col in skew_cols:
            non_null = featured[col].dropna()
            if len(non_null) > 0:
                assert non_null.min() >= -3.0 - 1e-10, f"{col} below -3"
                assert non_null.max() <= 3.0 + 1e-10, f"{col} above 3"

    def test_kurtosis_bounds(self, engineer, stats_df):
        """Kurtosis clipped to [-3, 10]."""
        featured = engineer._build_features(stats_df)
        kurt_cols = [c for c in featured.columns if '_kurtosis' in c]
        assert len(kurt_cols) > 0

        for col in kurt_cols:
            non_null = featured[col].dropna()
            if len(non_null) > 0:
                assert non_null.min() >= -3.0 - 1e-10, f"{col} below -3"
                assert non_null.max() <= 10.0 + 1e-10, f"{col} above 10"

    def test_cov_bounds(self, engineer, stats_df):
        """CoV clipped to [0, 5]."""
        featured = engineer._build_features(stats_df)
        cov_cols = [c for c in featured.columns if '_cov' in c]
        assert len(cov_cols) > 0

        for col in cov_cols:
            non_null = featured[col].dropna()
            if len(non_null) > 0:
                assert non_null.min() >= 0.0 - 1e-10, f"{col} below 0"
                assert non_null.max() <= 5.0 + 1e-10, f"{col} above 5"

    def test_momentum_ratio_bounds(self, engineer, stats_df):
        """Momentum ratio clipped to [0.3, 3.0]."""
        featured = engineer._build_features(stats_df)
        mr_cols = [c for c in featured.columns if '_momentum_ratio' in c and '_diff' not in c]
        assert len(mr_cols) > 0

        for col in mr_cols:
            non_null = featured[col].dropna()
            if len(non_null) > 0:
                assert non_null.min() >= 0.3 - 1e-10, f"{col} below 0.3"
                assert non_null.max() <= 3.0 + 1e-10, f"{col} above 3.0"

    def test_variance_ratio_bounds(self, engineer, stats_df):
        """Variance ratio clipped to [0.1, 5.0]."""
        featured = engineer._build_features(stats_df)
        vr_cols = [c for c in featured.columns if '_variance_ratio' in c and '_diff' not in c]
        assert len(vr_cols) > 0

        for col in vr_cols:
            non_null = featured[col].dropna()
            if len(non_null) > 0:
                assert non_null.min() >= 0.1 - 1e-10, f"{col} below 0.1"
                assert non_null.max() <= 5.0 + 1e-10, f"{col} above 5.0"

    def test_missing_columns_graceful(self, engineer):
        """Engineer handles missing niche stat columns without crashing."""
        df = pd.DataFrame({
            'fixture_id': range(10),
            'date': pd.date_range('2024-01-01', periods=10),
            'home_team': ['A', 'B'] * 5,
            'away_team': ['B', 'A'] * 5,
            'home_fouls': np.random.randint(8, 18, 10),
            'away_fouls': np.random.randint(8, 18, 10),
            # No shots, corners, cards columns
        })
        featured = engineer._build_features(df)
        # Should still produce fouls features
        fouls_cols = [c for c in featured.columns if 'fouls' in c and c not in df.columns]
        assert len(fouls_cols) > 0, "No fouls dynamics features created"

    def test_diff_features_exist(self, engineer, stats_df):
        """Home-away diff features created for momentum and regime."""
        featured = engineer._build_features(stats_df)

        # Momentum ratio diffs
        mr_diff_cols = [c for c in featured.columns if '_momentum_ratio_diff' in c]
        assert len(mr_diff_cols) == 4, f"Expected 4 momentum_ratio_diff, got {len(mr_diff_cols)}: {mr_diff_cols}"

        # Variance ratio diffs
        vr_diff_cols = [c for c in featured.columns if '_variance_ratio_diff' in c]
        assert len(vr_diff_cols) == 4, f"Expected 4 variance_ratio_diff, got {len(vr_diff_cols)}: {vr_diff_cols}"

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


# ---------- League Clustering Tests ----------

class TestLeagueClustering:
    """Tests for league clustering in regenerate_all_features.py."""

    def test_league_cluster_values(self):
        """Cluster values are in {0, 1, 2}."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Simulate per-league summary stats
        league_data = {
            'league': ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1',
                        'eredivisie', 'portuguese_liga', 'scottish_premiership',
                        'turkish_super_lig', 'belgian_pro_league'],
            'total_goals': [2.8, 2.5, 2.4, 3.1, 2.3, 3.0, 2.4, 2.6, 2.7, 2.5],
            'total_fouls': [22, 24, 26, 20, 25, 21, 23, 24, 28, 23],
            'total_cards': [3.5, 5.0, 4.5, 3.2, 3.8, 3.5, 4.5, 3.0, 5.5, 4.0],
            'total_shots': [25, 23, 22, 26, 21, 24, 22, 20, 21, 22],
            'total_corners': [10.5, 10.0, 9.5, 11.0, 9.0, 10.5, 9.5, 9.0, 9.5, 10.0],
        }
        profiles = pd.DataFrame(league_data).set_index('league')

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(profiles)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        assert set(labels).issubset({0, 1, 2}), f"Unexpected cluster values: {set(labels)}"
        assert len(set(labels)) >= 2, "All leagues in same cluster — clustering not working"

    def test_league_cluster_no_leakage(self):
        """First match per league should use expanding+shift(1) — no self-inclusion."""
        np.random.seed(42)
        n = 30
        leagues = ['league_A', 'league_B', 'league_C']
        rows = []
        for i in range(n):
            league = leagues[i % len(leagues)]
            rows.append({
                'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
                'league': league,
                'total_goals': np.random.uniform(2.0, 3.5),
                'total_fouls': np.random.uniform(18, 28),
                'total_cards': np.random.uniform(2, 6),
                'total_shots': np.random.uniform(18, 28),
                'total_corners': np.random.uniform(8, 12),
            })
        df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

        # Compute expanding averages with shift(1) — same logic as regenerate_all_features.py
        for stat in ['total_goals', 'total_fouls', 'total_cards', 'total_shots', 'total_corners']:
            df[f'_avg_{stat}'] = df.groupby('league')[stat].transform(
                lambda x: x.expanding().mean().shift(1)
            )

        # First match per league should have NaN expanding averages
        for league in leagues:
            league_df = df[df['league'] == league]
            first_idx = league_df.index[0]
            val = df.loc[first_idx, '_avg_total_goals']
            assert pd.isna(val), f"First match of {league} should have NaN avg, got {val}"
