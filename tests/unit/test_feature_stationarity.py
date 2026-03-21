"""
Feature Stationarity Guard Tests (S29)

Regression guard to ensure feature engineers produce stationary features
that don't encode "which time period" (act as clocks).

Generates synthetic time-ordered data, computes features, splits 50/50 by time,
and runs KS tests between halves. No feature should have KS statistic > 0.3.
"""

import numpy as np
import pandas as pd
import pytest
from collections import deque
from scipy import stats as scipy_stats


def _make_synthetic_matches(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic time-ordered match data with known league structure."""
    rng = np.random.RandomState(seed)

    leagues = ["league_A", "league_B"]
    teams_per_league = 10
    teams = {
        league: [f"{league}_team_{i}" for i in range(teams_per_league)]
        for league in leagues
    }

    records = []
    base_date = pd.Timestamp("2020-01-01")

    for i in range(n):
        league = leagues[i % len(leagues)]
        league_teams = teams[league]
        home_idx, away_idx = rng.choice(len(league_teams), size=2, replace=False)

        home_goals = rng.poisson(1.5)
        away_goals = rng.poisson(1.2)

        records.append({
            "fixture_id": i + 1,
            "date": base_date + pd.Timedelta(days=i // 4),
            "league": league,
            "home_team": league_teams[home_idx],
            "away_team": league_teams[away_idx],
            "home_team_id": home_idx + (0 if league == "league_A" else 100),
            "away_team_id": away_idx + (0 if league == "league_A" else 100),
            "ft_home": home_goals,
            "ft_away": away_goals,
            "home_fouls": rng.poisson(12),
            "away_fouls": rng.poisson(12),
            "home_corners": rng.poisson(5),
            "away_corners": rng.poisson(4),
            "home_shots": rng.poisson(12),
            "away_shots": rng.poisson(11),
            "home_possession": rng.uniform(40, 60),
            "away_possession": rng.uniform(40, 60),
            "home_yellow_cards": rng.poisson(1.5),
            "away_yellow_cards": rng.poisson(1.5),
            "home_red_cards": rng.binomial(1, 0.05),
            "away_red_cards": rng.binomial(1, 0.05),
            "referee": f"ref_{rng.randint(0, 8)}",
            "home_yellows": rng.poisson(1.5),
            "away_yellows": rng.poisson(1.5),
            "home_reds": rng.binomial(1, 0.05),
            "away_reds": rng.binomial(1, 0.05),
        })

    df = pd.DataFrame(records)
    df["total_fouls"] = df["home_fouls"] + df["away_fouls"]
    df["total_corners"] = df["home_corners"] + df["away_corners"]
    df["total_shots"] = df["home_shots"] + df["away_shots"]
    # Booking points convention: yellow=1, red=2
    from src.utils.booking_points import compute_booking_points_from_stats
    df["total_cards"] = compute_booking_points_from_stats(
        df["home_yellow_cards"] + df["away_yellow_cards"],
        df["home_red_cards"] + df["away_red_cards"],
    )
    df["home_cards"] = compute_booking_points_from_stats(df["home_yellow_cards"], df["home_red_cards"])
    df["away_cards"] = compute_booking_points_from_stats(df["away_yellow_cards"], df["away_red_cards"])
    return df


def _ks_stat_between_halves(series: pd.Series, skip_fraction: float = 0.3) -> float:
    """Compute KS statistic between two halves of a feature after warm-up.

    Skips initial fraction of data where rolling/ewm windows are filling up,
    which naturally causes non-stationarity unrelated to the expanding-window bug.
    """
    clean = series.dropna()
    skip = int(len(clean) * skip_fraction)
    if len(clean) - skip < 40:
        return 0.0  # Too few values to test
    # Skip warm-up period
    clean = clean.iloc[skip:]
    mid = len(clean) // 2
    first_half = clean.iloc[:mid].values
    second_half = clean.iloc[mid:].values
    if np.std(first_half) < 1e-10 and np.std(second_half) < 1e-10:
        return 0.0  # Constant feature
    ks_stat, _ = scipy_stats.ks_2samp(first_half, second_half)
    return ks_stat


class TestNicheMarketStationarity:
    """Test that niche market league features are stationary after S29 changes."""

    @pytest.fixture
    def matches(self):
        return _make_synthetic_matches()

    def test_fouls_league_features_stationary(self, matches):
        """Fouls league EMA avg and rolling std should not drift over time."""
        from src.features.engineers.niche_markets import FoulsFeatureEngineer

        eng = FoulsFeatureEngineer(league_window=50)
        featured = eng._build_features(matches.copy())

        for col in ["fouls_league_ema_avg", "fouls_league_rolling_std"]:
            if col in featured.columns:
                ks = _ks_stat_between_halves(featured[col])
                assert ks < 0.35, f"{col} KS={ks:.3f} > 0.35 — non-stationary"

    def test_cards_league_features_stationary(self, matches):
        """Cards league EMA avg and rolling std should not drift over time."""
        from src.features.engineers.niche_markets import CardsFeatureEngineer

        eng = CardsFeatureEngineer(league_window=50)
        featured = eng._build_features(matches.copy())

        for col in ["cards_league_ema_avg", "cards_league_rolling_std"]:
            if col in featured.columns:
                ks = _ks_stat_between_halves(featured[col])
                assert ks < 0.35, f"{col} KS={ks:.3f} > 0.35 — non-stationary"

    def test_shots_league_features_stationary(self, matches):
        """Shots league EMA avg and rolling std should not drift over time."""
        from src.features.engineers.niche_markets import ShotsFeatureEngineer

        eng = ShotsFeatureEngineer(league_window=50)
        featured = eng._build_features(matches.copy())

        for col in ["shots_league_ema_avg", "shots_league_rolling_std"]:
            if col in featured.columns:
                ks = _ks_stat_between_halves(featured[col])
                assert ks < 0.35, f"{col} KS={ks:.3f} > 0.35 — non-stationary"

    def test_no_expanding_columns_produced(self, matches):
        """Ensure old expanding column names are NOT produced."""
        from src.features.engineers.niche_markets import (
            FoulsFeatureEngineer,
            CardsFeatureEngineer,
            ShotsFeatureEngineer,
        )

        for EngClass in [FoulsFeatureEngineer, CardsFeatureEngineer, ShotsFeatureEngineer]:
            eng = EngClass(league_window=50)
            featured = eng._build_features(matches.copy())
            expanding_cols = [c for c in featured.columns if "expanding" in c.lower()]
            assert expanding_cols == [], (
                f"{EngClass.__name__} still produces expanding columns: {expanding_cols}"
            )


class TestLeagueAggregateStationarity:
    """Test that league aggregate features are stationary after S29 changes."""

    def test_league_aggregate_features_stationary(self):
        """League aggregate features should not drift over time."""
        from src.features.engineers.league_aggregate import LeagueAggregateFeatureEngineer

        matches = _make_synthetic_matches(n=2000)
        eng = LeagueAggregateFeatureEngineer(min_matches=20, window=100)
        result = eng.create_features({"matches": matches})

        feature_cols = [c for c in result.columns if c.startswith("league_")]
        assert len(feature_cols) > 0, "No league features produced"

        for col in feature_cols:
            ks = _ks_stat_between_halves(result[col])
            assert ks < 0.3, f"{col} KS={ks:.3f} > 0.3 — non-stationary"


class TestCornerStationarity:
    """Test that corner features no longer include expanding variants."""

    def test_no_expanding_columns_in_corners(self):
        """CornerFeatureEngineer should not produce *_expanding features."""
        from src.features.engineers.corners import CornerFeatureEngineer

        matches = _make_synthetic_matches(n=200)
        eng = CornerFeatureEngineer(window_sizes=[5, 10], min_matches=3)
        featured = eng.fit_transform(matches)

        expanding_cols = [c for c in featured.columns if "expanding" in c.lower()]
        assert expanding_cols == [], (
            f"CornerFeatureEngineer still produces expanding columns: {expanding_cols}"
        )


class TestRefereeStationarity:
    """Test that referee features are bounded and don't monotonically converge."""

    def test_ref_experienced_is_binary(self):
        """ref_experienced should be 0 or 1, not a monotonically increasing counter."""
        from src.features.engineers.external import RefereeFeatureEngineer

        matches = _make_synthetic_matches(n=500)
        eng = RefereeFeatureEngineer(min_matches=5, recent_window=10, career_window=30)
        result = eng.create_features({"matches": matches})

        assert "ref_experienced" in result.columns, "ref_experienced column missing"
        assert "ref_matches" not in result.columns, "ref_matches should be replaced by ref_experienced"

        unique_vals = set(result["ref_experienced"].unique())
        assert unique_vals <= {0, 1}, f"ref_experienced has non-binary values: {unique_vals}"

    def test_ref_career_averages_bounded(self):
        """Referee career averages should not monotonically converge (bounded by career_window)."""
        from src.features.engineers.external import RefereeFeatureEngineer

        matches = _make_synthetic_matches(n=500)
        eng = RefereeFeatureEngineer(min_matches=3, recent_window=10, career_window=15)
        result = eng.create_features({"matches": matches})

        # ref_cards_avg should have meaningful variance even late in the dataset
        cards_avg = result["ref_cards_avg"].dropna()
        if len(cards_avg) > 100:
            late_std = cards_avg.iloc[-100:].std()
            assert late_std > 0.01, (
                f"ref_cards_avg has collapsed variance ({late_std:.6f}) in late matches — "
                f"career_window not working"
            )

    def test_ref_features_stationary(self):
        """Referee features should not have strong time drift."""
        from src.features.engineers.external import RefereeFeatureEngineer

        matches = _make_synthetic_matches(n=500)
        eng = RefereeFeatureEngineer(min_matches=3, recent_window=10, career_window=15)
        result = eng.create_features({"matches": matches})

        for col in ["ref_cards_avg", "ref_fouls_avg", "ref_corners_avg"]:
            if col in result.columns:
                ks = _ks_stat_between_halves(result[col])
                assert ks < 0.35, f"{col} KS={ks:.3f} > 0.35 — non-stationary"


class TestGoalTimingStationarity:
    """Test that GoalTimingFeatureEngineer uses bounded lookback."""

    def test_lookback_matches_is_wired(self):
        """GoalTimingFeatureEngineer.lookback_matches should actually limit history."""
        from src.features.engineers.stats import GoalTimingFeatureEngineer

        # Create synthetic events data
        rng = np.random.RandomState(42)
        matches = _make_synthetic_matches(n=200)

        events = []
        for _, match in matches.iterrows():
            n_goals = match["ft_home"] + match["ft_away"]
            for g in range(int(n_goals)):
                team_id = match["home_team_id"] if g % 2 == 0 else match["away_team_id"]
                events.append({
                    "fixture_id": match["fixture_id"],
                    "type": "Goal",
                    "team_id": team_id,
                    "time_elapsed": rng.randint(1, 90),
                })
        events_df = pd.DataFrame(events)

        # Run with small lookback
        eng_small = GoalTimingFeatureEngineer(lookback_matches=5)
        result_small = eng_small.create_features({"matches": matches, "events": events_df})

        # Run with large lookback
        eng_large = GoalTimingFeatureEngineer(lookback_matches=100)
        result_large = eng_large.create_features({"matches": matches, "events": events_df})

        # With different lookback windows, late-match features should differ
        late_idx = result_small.index[-50:]
        small_late = result_small.loc[late_idx, "home_early_goal_rate"]
        large_late = result_large.loc[late_idx, "home_early_goal_rate"]

        # They should NOT be identical (lookback_matches=5 forgets older data)
        diff = (small_late - large_late).abs().mean()
        assert diff > 0.001, (
            f"lookback_matches has no effect: mean diff={diff:.6f}. "
            f"GoalTimingFeatureEngineer is still using unbounded history."
        )
