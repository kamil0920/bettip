"""Unit tests for betting strategies module."""
import pytest
import numpy as np
import pandas as pd

from src.ml.betting_strategies import (
    StrategyConfig,
    BettingStrategy,
    AsianHandicapStrategy,
    BTTSStrategy,
    AwayWinStrategy,
    HomeWinStrategy,
    Over25Strategy,
    Under25Strategy,
    CornersStrategy,
    CardsStrategy,
    ShotsStrategy,
    FoulsStrategy,
    get_strategy,
    register_strategy,
    STRATEGY_REGISTRY,
)


class TestStrategyConfig:
    """Tests for StrategyConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StrategyConfig()
        assert config.enabled is True
        assert config.approach == "classification"
        assert config.probability_threshold == 0.5
        assert config.edge_threshold == 0.3
        assert config.kelly_fraction == 0.25
        assert config.max_stake_fraction == 0.05

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StrategyConfig(
            enabled=False,
            probability_threshold=0.6,
            kelly_fraction=0.5
        )
        assert config.enabled is False
        assert config.probability_threshold == 0.6
        assert config.kelly_fraction == 0.5


class TestStrategyRegistry:
    """Tests for strategy registry."""

    def test_all_strategies_registered(self):
        """Test all expected strategies are in registry."""
        expected = [
            "asian_handicap",
            "btts",
            "away_win",
            "home_win",
            "over25",
            "under25",
        ]
        for name in expected:
            assert name in STRATEGY_REGISTRY

    def test_get_strategy(self):
        """Test getting strategy by name."""
        strategy = get_strategy("btts")
        assert isinstance(strategy, BTTSStrategy)

    def test_get_strategy_with_config(self):
        """Test getting strategy with custom config."""
        config = StrategyConfig(probability_threshold=0.7)
        strategy = get_strategy("btts", config)
        assert strategy.config.probability_threshold == 0.7

    def test_get_strategy_invalid_name(self):
        """Test getting invalid strategy raises error."""
        with pytest.raises(ValueError) as exc_info:
            get_strategy("invalid_strategy")
        assert "Unknown strategy" in str(exc_info.value)

    def test_register_strategy(self):
        """Test registering new strategy."""
        class CustomStrategy(BettingStrategy):
            @property
            def name(self):
                return "custom"

            @property
            def is_regression(self):
                return False

            @property
            def default_odds_column(self):
                return "custom_odds"

            @property
            def bet_side(self):
                return "custom"

            def create_target(self, df):
                return df, "target"

            def evaluate(self, predictions, y_test, odds_test, df_test=None):
                return []

        register_strategy("custom", CustomStrategy)
        assert "custom" in STRATEGY_REGISTRY

        # Cleanup
        del STRATEGY_REGISTRY["custom"]

    def test_register_invalid_strategy(self):
        """Test registering non-strategy class raises error."""
        class NotAStrategy:
            pass

        with pytest.raises(TypeError):
            register_strategy("invalid", NotAStrategy)


class TestAsianHandicapStrategy:
    """Tests for Asian Handicap strategy."""

    @pytest.fixture
    def strategy(self):
        return AsianHandicapStrategy()

    def test_properties(self, strategy):
        """Test strategy properties."""
        assert strategy.name == "asian_handicap"
        assert strategy.is_regression is True
        assert strategy.default_odds_column == "avg_ah_away"
        assert strategy.bet_side == "away"

    def test_create_target(self, strategy):
        """Test target creation."""
        df = pd.DataFrame({
            "ah_line": [-1.5, -2.0, None],
            "avg_ah_home": [1.9, 1.85, None],
            "goal_difference": [2, -1, 0]
        })
        df_filtered, target_col = strategy.create_target(df)

        assert target_col == "goal_margin"
        assert len(df_filtered) == 2
        assert "goal_margin" in df_filtered.columns

    def test_format_bet_side(self, strategy):
        """Test bet side formatting with handicap line."""
        row = pd.Series({"ah_line": -1.5})
        result = strategy._format_bet_side(row)
        assert result == "away -1.50"


class TestBTTSStrategy:
    """Tests for BTTS strategy."""

    @pytest.fixture
    def strategy(self):
        return BTTSStrategy()

    def test_properties(self, strategy):
        """Test strategy properties."""
        assert strategy.name == "btts"
        assert strategy.is_regression is False
        assert strategy.default_odds_column == "btts_yes_avg"
        assert strategy.bet_side == "yes"

    def test_create_target_with_btts_column(self, strategy):
        """Test target creation when btts column exists."""
        df = pd.DataFrame({
            "btts": [1, 0, 1],
            "home_goals": [2, 0, 1],
            "away_goals": [1, 0, 2]
        })
        df_filtered, target_col = strategy.create_target(df)

        assert target_col == "btts"
        assert len(df_filtered) == 3

    def test_create_target_calculates_btts(self, strategy):
        """Test target creation calculates btts from goals."""
        df = pd.DataFrame({
            "home_goals": [2, 0, 1, 3],
            "away_goals": [1, 0, 2, 0]
        })
        df_filtered, target_col = strategy.create_target(df)

        assert target_col == "btts"
        expected_btts = [1, 0, 1, 0]
        assert df_filtered["btts"].tolist() == expected_btts

    def test_create_target_handles_nan_goals(self, strategy):
        """Test target creation filters out NaN goal values."""
        df = pd.DataFrame({
            "home_goals": [2, None, 1, 3, None],
            "away_goals": [1, 0, None, 0, None]
        })
        df_filtered, target_col = strategy.create_target(df)

        assert target_col == "btts"
        # Only rows 0 and 3 have both goals non-null
        assert len(df_filtered) == 2
        assert df_filtered["btts"].tolist() == [1, 0]

    def test_create_target_handles_existing_nan_btts(self, strategy):
        """Test target creation handles existing btts column with NaN."""
        df = pd.DataFrame({
            "btts": [1, None, 0, None, 1],
            "home_goals": [2, 0, 1, 3, 2],
            "away_goals": [1, 0, 2, 0, 1]
        })
        df_filtered, target_col = strategy.create_target(df)

        assert target_col == "btts"
        assert len(df_filtered) == 3
        assert df_filtered["btts"].tolist() == [1, 0, 1]


class TestAwayWinStrategy:
    """Tests for Away Win strategy."""

    @pytest.fixture
    def strategy(self):
        return AwayWinStrategy()

    def test_properties(self, strategy):
        """Test strategy properties."""
        assert strategy.name == "away_win"
        assert strategy.is_regression is False
        assert strategy.default_odds_column == "avg_away_open"
        assert strategy.bet_side == "away win"

    def test_create_target(self, strategy):
        """Test target creation."""
        df = pd.DataFrame({
            "avg_away_open": [3.5, 4.0, None],
            "away_win": [1, 0, 1]
        })
        df_filtered, target_col = strategy.create_target(df)

        assert target_col == "target"
        assert len(df_filtered) == 2


class TestHomeWinStrategy:
    """Tests for Home Win strategy."""

    @pytest.fixture
    def strategy(self):
        return HomeWinStrategy()

    def test_properties(self, strategy):
        """Test strategy properties."""
        assert strategy.name == "home_win"
        assert strategy.is_regression is False
        assert strategy.default_odds_column == "avg_home_open"
        assert strategy.bet_side == "home win"


class TestTotalsStrategies:
    """Tests for Over/Under strategies."""

    def test_over25_properties(self):
        """Test Over 2.5 strategy properties."""
        strategy = Over25Strategy()
        assert strategy.name == "over25"
        assert strategy.total_line == 2.5
        assert strategy.is_over is True
        assert strategy.bet_side == "over 2.5"

    def test_under25_properties(self):
        """Test Under 2.5 strategy properties."""
        strategy = Under25Strategy()
        assert strategy.name == "under25"
        assert strategy.total_line == 2.5
        assert strategy.is_over is False
        assert strategy.bet_side == "under 2.5"

    def test_over25_create_target(self):
        """Test Over 2.5 target creation."""
        strategy = Over25Strategy()
        df = pd.DataFrame({
            "avg_over25": [1.8, 2.0, None],
            "total_goals": [3, 2, 4]
        })
        df_filtered, target_col = strategy.create_target(df)

        assert len(df_filtered) == 2
        assert df_filtered["target"].tolist() == [1, 0]

    def test_under25_create_target(self):
        """Test Under 2.5 target creation."""
        strategy = Under25Strategy()
        df = pd.DataFrame({
            "avg_under25": [1.8, 2.0, None],
            "total_goals": [2, 4, 1]
        })
        df_filtered, target_col = strategy.create_target(df)

        assert len(df_filtered) == 2
        assert df_filtered["target"].tolist() == [1, 0]


class TestNicheStrategies:
    """Tests for niche market strategies."""

    def test_corners_strategy_properties(self):
        """Test corners strategy properties."""
        strategy = CornersStrategy()
        assert strategy.name == "corners"
        assert strategy.stat_column == "total_corners"
        assert strategy.is_regression is False

    def test_corners_strategy_create_target(self):
        """Test corners target creation."""
        strategy = CornersStrategy()
        df = pd.DataFrame({
            "total_corners": [12, 8, 15, None],
        })
        df_filtered, target_col = strategy.create_target(df)

        assert target_col == "target"
        assert len(df_filtered) == 3
        # Over 10.5: 12 > 10.5 = 1, 8 > 10.5 = 0, 15 > 10.5 = 1
        assert df_filtered["target"].tolist() == [1, 0, 1]

    def test_cards_strategy_properties(self):
        """Test cards strategy properties."""
        strategy = CardsStrategy()
        assert strategy.name == "cards"
        assert strategy.stat_column == "total_cards"

    def test_shots_strategy_properties(self):
        """Test shots strategy properties."""
        strategy = ShotsStrategy()
        assert strategy.name == "shots"
        assert strategy.stat_column == "total_shots"

    def test_fouls_strategy_properties(self):
        """Test fouls strategy properties."""
        strategy = FoulsStrategy()
        assert strategy.name == "fouls"
        assert strategy.stat_column == "total_fouls"
        assert strategy.ref_stat_column == "ref_fouls_avg"

    def test_niche_strategies_in_registry(self):
        """Test all niche strategies are registered."""
        assert "corners" in STRATEGY_REGISTRY
        assert "cards" in STRATEGY_REGISTRY
        assert "shots" in STRATEGY_REGISTRY
        assert "fouls" in STRATEGY_REGISTRY

    def test_get_niche_strategy(self):
        """Test getting niche strategy by name."""
        strategy = get_strategy("corners")
        assert isinstance(strategy, CornersStrategy)

    def test_corners_strategy_missing_column_raises(self):
        """Test corners strategy raises ValueError when column missing."""
        strategy = CornersStrategy()
        df = pd.DataFrame({
            "home_team": ["A", "B"],
            "away_team": ["C", "D"],
        })
        with pytest.raises(ValueError) as exc_info:
            strategy.create_target(df)
        assert "total_corners" in str(exc_info.value)

    def test_cards_strategy_missing_column_raises(self):
        """Test cards strategy raises ValueError when column missing."""
        strategy = CardsStrategy()
        df = pd.DataFrame({"home_team": ["A"]})
        with pytest.raises(ValueError) as exc_info:
            strategy.create_target(df)
        assert "total_cards" in str(exc_info.value)

    def test_shots_strategy_missing_column_raises(self):
        """Test shots strategy raises ValueError when column missing."""
        strategy = ShotsStrategy()
        df = pd.DataFrame({"home_team": ["A"]})
        with pytest.raises(ValueError) as exc_info:
            strategy.create_target(df)
        assert "total_shots" in str(exc_info.value)

    def test_fouls_strategy_missing_column_raises(self):
        """Test fouls strategy raises ValueError when column missing."""
        strategy = FoulsStrategy()
        df = pd.DataFrame({"home_team": ["A"]})
        with pytest.raises(ValueError) as exc_info:
            strategy.create_target(df)
        assert "total_fouls" in str(exc_info.value)


class TestCreateRecommendation:
    """Tests for recommendation creation."""

    @pytest.fixture
    def strategy(self):
        config = StrategyConfig(
            min_edge=0.05,
            kelly_fraction=0.25,
            max_stake_fraction=0.05
        )
        return BTTSStrategy(config)

    def test_create_recommendation_meets_criteria(self, strategy):
        """Test recommendation creation when criteria met."""
        row = pd.Series({
            "fixture_id": "12345",
            "date": "2026-01-15",
            "home_team_name": "Team A",
            "away_team_name": "Team B",
            "league": "premier_league",
            "btts_yes_avg": 1.8  # Implied prob ~55%
        })
        prediction = 0.65  # Our prob 65%, edge = 65% - 55% = 10%

        rec = strategy.create_recommendation(row, prediction, bankroll=1000)

        assert rec is not None
        assert rec["fixture_id"] == "12345"
        assert rec["bet_type"] == "btts"
        assert rec["probability"] == 0.65
        assert rec["edge"] > 0.05
        assert rec["recommended_stake"] > 0
        assert rec["recommended_stake"] <= 50  # Max 5% of 1000

    def test_create_recommendation_below_min_edge(self, strategy):
        """Test no recommendation when edge below minimum."""
        row = pd.Series({
            "btts_yes_avg": 1.8  # Implied prob ~55%
        })
        prediction = 0.56  # Edge ~1%, below min_edge

        rec = strategy.create_recommendation(row, prediction)
        assert rec is None

    def test_create_recommendation_invalid_odds(self, strategy):
        """Test no recommendation with invalid odds."""
        row = pd.Series({
            "btts_yes_avg": 0.9  # Invalid odds <= 1
        })
        rec = strategy.create_recommendation(row, 0.7)
        assert rec is None

    def test_create_recommendation_missing_odds(self, strategy):
        """Test no recommendation with missing odds."""
        row = pd.Series({
            "btts_yes_avg": None
        })
        rec = strategy.create_recommendation(row, 0.7)
        assert rec is None


class TestKellyCalculation:
    """Tests for Kelly criterion calculations."""

    def test_kelly_positive_edge(self):
        """Test Kelly stake with positive edge."""
        config = StrategyConfig(kelly_fraction=1.0, max_stake_fraction=0.10)
        strategy = BTTSStrategy(config)

        row = pd.Series({"btts_yes_avg": 2.0})  # Implied 50%
        rec = strategy.create_recommendation(row, 0.6, bankroll=1000)  # 60% prob

        # Full Kelly = (0.6 * 2 - 1) / (2 - 1) = 0.2 / 1 = 0.2
        assert rec is not None
        assert rec["kelly_fraction"] <= 0.10  # Capped

    def test_kelly_fractional(self):
        """Test fractional Kelly (quarter Kelly)."""
        config = StrategyConfig(kelly_fraction=0.25, max_stake_fraction=0.10)
        strategy = BTTSStrategy(config)

        row = pd.Series({"btts_yes_avg": 2.0})
        rec = strategy.create_recommendation(row, 0.7, bankroll=1000)

        # Full Kelly = (0.7 * 2 - 1) / 1 = 0.4
        # Quarter Kelly = 0.4 * 0.25 = 0.10
        assert rec is not None
        assert rec["kelly_fraction"] <= 0.10


class TestBootstrapROI:
    """Tests for bootstrap ROI calculation."""

    def test_calc_roi_bootstrap_profitable(self):
        """Test bootstrap ROI for profitable scenario."""
        strategy = BTTSStrategy()

        # All wins at odds 2.0
        predictions = np.ones(100)
        actuals = np.ones(100)
        odds = np.full(100, 2.0)

        roi, ci_low, ci_high, p_profit = strategy.calc_roi_bootstrap(
            predictions, actuals, odds, n_boot=100
        )

        assert roi > 0  # Profitable
        assert p_profit == 1.0  # 100% probability of profit
        assert ci_low > 0

    def test_calc_roi_bootstrap_unprofitable(self):
        """Test bootstrap ROI for unprofitable scenario."""
        strategy = BTTSStrategy()

        # All losses
        predictions = np.ones(100)
        actuals = np.zeros(100)
        odds = np.full(100, 2.0)

        roi, ci_low, ci_high, p_profit = strategy.calc_roi_bootstrap(
            predictions, actuals, odds, n_boot=100
        )

        assert roi < 0  # Unprofitable
        assert p_profit == 0.0
        assert ci_high < 0

    def test_calc_roi_bootstrap_no_bets(self):
        """Test bootstrap ROI when no bets placed."""
        strategy = BTTSStrategy()

        predictions = np.zeros(100)  # No bets
        actuals = np.ones(100)
        odds = np.full(100, 2.0)

        roi, ci_low, ci_high, p_profit = strategy.calc_roi_bootstrap(
            predictions, actuals, odds
        )

        assert roi == 0
        assert p_profit == 0
