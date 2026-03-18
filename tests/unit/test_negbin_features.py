"""Tests for NegBin features and new feature engineers (Elo SD, WSI).

Validates:
1. negbin_over_probability: NB > Poisson in tails, NB = Poisson when d=1.0
2. NegBin features exist in niche engineer output
3. Elo SD features computed correctly
4. WSI features computed correctly
5. No data leakage in any new features
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from src.odds.count_distribution import (
    DISPERSION_RATIOS,
    negbin_over_probability,
    overdispersed_cdf,
)


class TestNegBinOverProbability:
    """Tests for negbin_over_probability() convenience function."""

    def test_complement_of_cdf(self):
        """negbin_over_probability should equal 1 - CDF."""
        lam, line = 24.5, 24.5
        result = negbin_over_probability(lam, line, "fouls")
        expected = 1.0 - overdispersed_cdf(line, lam, "fouls")
        assert result == pytest.approx(expected)

    def test_negbin_wider_tails_than_poisson(self):
        """NegBin P(X > line) should differ from Poisson for overdispersed stats."""
        lam = 24.5
        line = 28.5  # Far right tail
        negbin_prob = negbin_over_probability(lam, line, "fouls")
        poisson_prob = 1.0 - poisson.cdf(line, lam)
        # NegBin has heavier tails → higher prob in right tail
        assert negbin_prob > poisson_prob

    def test_negbin_equals_poisson_when_d_is_one(self):
        """When dispersion=1.0, NegBin should match Poisson exactly."""
        lam = 10.0
        line = 12.5
        negbin_prob = negbin_over_probability(lam, line, "unknown_stat")
        poisson_prob = 1.0 - poisson.cdf(line, lam)
        assert negbin_prob == pytest.approx(poisson_prob)

    def test_array_input(self):
        """Should handle array inputs correctly."""
        lam = np.array([20.0, 25.0, 30.0])
        line = 24.5
        result = negbin_over_probability(lam, line, "fouls")
        assert result.shape == (3,)
        # Higher expected count → higher probability of exceeding line
        assert result[0] < result[1] < result[2]

    def test_probability_bounds(self):
        """Output should always be in [0, 1]."""
        for lam in [5.0, 15.0, 25.0, 40.0]:
            for line in [10.5, 20.5, 30.5]:
                prob = negbin_over_probability(lam, line, "cards")
                assert 0.0 <= prob <= 1.0


class TestEloSDFeatures:
    """Tests for Elo standard deviation features."""

    @pytest.fixture
    def elo_engineer(self):
        from src.features.engineers.ratings import ELORatingFeatureEngineer
        return ELORatingFeatureEngineer(k_factor=32.0, home_advantage=100.0, sd_window=10)

    @pytest.fixture
    def sample_matches(self):
        """Create minimal match data for testing."""
        np.random.seed(42)
        n_matches = 30
        teams = [1, 2, 3, 4]
        rows = []
        for i in range(n_matches):
            home = teams[i % len(teams)]
            away = teams[(i + 1) % len(teams)]
            rows.append({
                'fixture_id': i + 1,
                'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
                'home_team_id': home,
                'away_team_id': away,
                'ft_home': np.random.randint(0, 4),
                'ft_away': np.random.randint(0, 3),
            })
        return pd.DataFrame(rows)

    def test_elo_sd_columns_exist(self, elo_engineer, sample_matches):
        """Elo SD features should appear in output."""
        result = elo_engineer.create_features({'matches': sample_matches})
        assert 'home_elo_sd' in result.columns
        assert 'away_elo_sd' in result.columns
        assert 'elo_sd_diff' in result.columns

    def test_elo_sd_nan_for_early_matches(self, elo_engineer, sample_matches):
        """First few matches should have NaN SD (< 3 deltas)."""
        result = elo_engineer.create_features({'matches': sample_matches})
        # First match for each team should be NaN
        assert pd.isna(result.iloc[0]['home_elo_sd'])

    def test_elo_sd_nonneg_after_warmup(self, elo_engineer, sample_matches):
        """After warmup period, SD should be non-negative."""
        result = elo_engineer.create_features({'matches': sample_matches})
        valid = result['home_elo_sd'].dropna()
        assert (valid >= 0).all()

    def test_elo_sd_no_future_leakage(self, elo_engineer, sample_matches):
        """SD should only use deltas from BEFORE the current match."""
        result = elo_engineer.create_features({'matches': sample_matches})
        # Run on first 15 matches vs all 30 — first 15 rows should match
        partial = elo_engineer.create_features({
            'matches': sample_matches.iloc[:15].copy()
        })
        for col in ['home_elo_sd', 'away_elo_sd']:
            for i in range(15):
                a = result.iloc[i][col]
                b = partial.iloc[i][col]
                if pd.isna(a) and pd.isna(b):
                    continue
                assert a == pytest.approx(b, abs=1e-6), f"Row {i}, col {col}: {a} != {b}"


class TestWSIFeatures:
    """Tests for Weighted Streak Index features."""

    @pytest.fixture
    def streak_engineer(self):
        from src.features.engineers.form import StreakFeatureEngineer
        return StreakFeatureEngineer(wsi_window=6)

    @pytest.fixture
    def sample_matches(self):
        np.random.seed(42)
        n_matches = 30
        teams = [1, 2, 3, 4]
        rows = []
        for i in range(n_matches):
            home = teams[i % len(teams)]
            away = teams[(i + 1) % len(teams)]
            rows.append({
                'fixture_id': i + 1,
                'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
                'home_team_id': home,
                'away_team_id': away,
                'ft_home': np.random.randint(0, 4),
                'ft_away': np.random.randint(0, 3),
            })
        return pd.DataFrame(rows)

    def test_wsi_columns_exist(self, streak_engineer, sample_matches):
        """WSI features should appear in output."""
        result = streak_engineer.create_features({'matches': sample_matches})
        assert 'home_weighted_streak_index' in result.columns
        assert 'away_weighted_streak_index' in result.columns
        assert 'weighted_streak_diff' in result.columns

    def test_wsi_nan_for_first_match(self, streak_engineer, sample_matches):
        """First match for each team should have NaN WSI (no history)."""
        result = streak_engineer.create_features({'matches': sample_matches})
        assert pd.isna(result.iloc[0]['home_weighted_streak_index'])

    def test_wsi_bounded(self, streak_engineer, sample_matches):
        """WSI should be bounded in [-1, 1]."""
        result = streak_engineer.create_features({'matches': sample_matches})
        valid = result['home_weighted_streak_index'].dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()

    def test_wsi_all_wins_positive(self):
        """Team with all wins should have WSI close to +1."""
        from src.features.engineers.form import StreakFeatureEngineer
        eng = StreakFeatureEngineer(wsi_window=6)
        # Create matches where team 1 always wins at home
        rows = []
        for i in range(10):
            rows.append({
                'fixture_id': i + 1,
                'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
                'home_team_id': 1,
                'away_team_id': 2,
                'ft_home': 3,
                'ft_away': 0,
            })
        matches = pd.DataFrame(rows)
        result = eng.create_features({'matches': matches})
        # After 6 matches, WSI should be close to 1.0
        wsi = result.iloc[8]['home_weighted_streak_index']
        assert wsi > 0.9

    def test_wsi_no_future_leakage(self, streak_engineer, sample_matches):
        """WSI should only use results from BEFORE the current match."""
        result = streak_engineer.create_features({'matches': sample_matches})
        partial = streak_engineer.create_features({
            'matches': sample_matches.iloc[:15].copy()
        })
        for col in ['home_weighted_streak_index', 'away_weighted_streak_index']:
            for i in range(15):
                a = result.iloc[i][col]
                b = partial.iloc[i][col]
                if pd.isna(a) and pd.isna(b):
                    continue
                assert a == pytest.approx(b, abs=1e-6), f"Row {i}, col {col}: {a} != {b}"


class TestNegBinNicheFeatures:
    """Tests for NegBin features in niche market engineers."""

    def test_fouls_negbin_columns(self, tmp_path):
        """FoulsFeatureEngineer should produce NegBin feature columns."""
        from src.features.engineers.niche_markets import FoulsFeatureEngineer
        eng = FoulsFeatureEngineer(ema_span=5, min_matches=2)

        # Create minimal data that _build_features expects
        df = pd.DataFrame({
            'fixture_id': range(20),
            'date': pd.date_range('2024-01-01', periods=20),
            'home_team': ['A'] * 10 + ['B'] * 10,
            'away_team': ['B'] * 10 + ['A'] * 10,
            'home_fouls': np.random.randint(8, 18, 20),
            'away_fouls': np.random.randint(8, 18, 20),
            'league': ['test'] * 20,
        })
        result = eng._build_features(df)

        assert 'negbin_fouls_over_245_prob' in result.columns
        assert 'negbin_fouls_expected_std' in result.columns
        assert 'fouls_tail_weight' in result.columns

    def test_shots_negbin_columns(self, tmp_path):
        """ShotsFeatureEngineer should produce NegBin feature columns."""
        from src.features.engineers.niche_markets import ShotsFeatureEngineer
        eng = ShotsFeatureEngineer(ema_span=5, min_matches=2)

        df = pd.DataFrame({
            'fixture_id': range(20),
            'date': pd.date_range('2024-01-01', periods=20),
            'home_team': ['A'] * 10 + ['B'] * 10,
            'away_team': ['B'] * 10 + ['A'] * 10,
            'home_shots': np.random.randint(8, 18, 20),
            'away_shots': np.random.randint(8, 18, 20),
            'home_shots_on_target': np.random.randint(2, 8, 20),
            'away_shots_on_target': np.random.randint(2, 8, 20),
            'league': ['test'] * 20,
        })
        result = eng._build_features(df)

        assert 'negbin_shots_over_245_prob' in result.columns
        assert 'negbin_shots_expected_std' in result.columns
        assert 'shots_tail_weight' in result.columns

    def test_cards_negbin_columns(self):
        """CardsFeatureEngineer should produce NegBin feature columns."""
        from src.features.engineers.niche_markets import CardsFeatureEngineer
        eng = CardsFeatureEngineer(ema_span=5, min_matches=2)

        df = pd.DataFrame({
            'fixture_id': range(20),
            'date': pd.date_range('2024-01-01', periods=20),
            'home_team': ['A'] * 10 + ['B'] * 10,
            'away_team': ['B'] * 10 + ['A'] * 10,
            'home_cards': np.random.randint(0, 5, 20).astype(float),
            'away_cards': np.random.randint(0, 5, 20).astype(float),
            'total_cards': np.random.randint(2, 8, 20).astype(float),
            'league': ['test'] * 20,
        })
        result = eng._build_features(df)

        assert 'negbin_cards_over_45_prob' in result.columns
        assert 'negbin_cards_expected_std' in result.columns
        assert 'cards_tail_weight' in result.columns

    def test_negbin_prob_bounded(self):
        """NegBin probability features should be in [0, 1]."""
        from src.features.engineers.niche_markets import FoulsFeatureEngineer
        eng = FoulsFeatureEngineer(ema_span=5, min_matches=2)

        df = pd.DataFrame({
            'fixture_id': range(30),
            'date': pd.date_range('2024-01-01', periods=30),
            'home_team': ['A'] * 15 + ['B'] * 15,
            'away_team': ['B'] * 15 + ['A'] * 15,
            'home_fouls': np.random.randint(8, 18, 30),
            'away_fouls': np.random.randint(8, 18, 30),
            'league': ['test'] * 30,
        })
        result = eng._build_features(df)
        prob = result['negbin_fouls_over_245_prob'].dropna()
        assert (prob >= 0).all()
        assert (prob <= 1).all()

    def test_tail_weight_bounded(self):
        """Tail weight should be in [0, 1]."""
        from src.features.engineers.niche_markets import FoulsFeatureEngineer
        eng = FoulsFeatureEngineer(ema_span=5, min_matches=2)

        df = pd.DataFrame({
            'fixture_id': range(30),
            'date': pd.date_range('2024-01-01', periods=30),
            'home_team': ['A'] * 15 + ['B'] * 15,
            'away_team': ['B'] * 15 + ['A'] * 15,
            'home_fouls': np.random.randint(8, 18, 30),
            'away_fouls': np.random.randint(8, 18, 30),
            'league': ['test'] * 30,
        })
        result = eng._build_features(df)
        tw = result['fouls_tail_weight'].dropna()
        assert (tw >= 0).all()
        assert (tw <= 1).all()
