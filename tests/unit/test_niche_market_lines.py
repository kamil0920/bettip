"""Unit tests for niche market multi-line support."""
import pytest
import numpy as np
import pandas as pd

from src.ml.betting_strategies import (
    CardsStrategy,
    CornersStrategy,
    ShotsStrategy,
    FoulsStrategy,
    NicheMarketStrategy,
    get_strategy,
    STRATEGY_REGISTRY,
    NICHE_LINE_LOOKUP,
    BASE_MARKET_MAP,
)


class TestStrategyParameterization:
    """Test that niche strategies accept and use custom lines."""

    def test_cards_default_line(self):
        s = CardsStrategy()
        assert s.line == 4.5
        assert s.name == "cards"

    def test_cards_custom_line_35(self):
        s = CardsStrategy(line=3.5)
        assert s.line == 3.5
        assert s.name == "cards_over_35"

    def test_cards_custom_line_55(self):
        s = CardsStrategy(line=5.5)
        assert s.line == 5.5
        assert s.name == "cards_over_55"

    def test_cards_custom_line_65(self):
        s = CardsStrategy(line=6.5)
        assert s.line == 6.5
        assert s.name == "cards_over_65"

    def test_corners_default_line(self):
        s = CornersStrategy()
        assert s.line == 9.5
        assert s.name == "corners"

    def test_corners_custom_line_85(self):
        s = CornersStrategy(line=8.5)
        assert s.line == 8.5
        assert s.name == "corners_over_85"

    def test_corners_custom_line_105(self):
        s = CornersStrategy(line=10.5)
        assert s.line == 10.5
        assert s.name == "corners_over_105"

    def test_corners_custom_line_115(self):
        s = CornersStrategy(line=11.5)
        assert s.line == 11.5
        assert s.name == "corners_over_115"

    def test_shots_default_line(self):
        s = ShotsStrategy()
        assert s.line == 24.5
        assert s.name == "shots"

    def test_shots_custom_line_225(self):
        s = ShotsStrategy(line=22.5)
        assert s.line == 22.5
        assert s.name == "shots_over_225"

    def test_fouls_default_line(self):
        s = FoulsStrategy()
        assert s.line == 24.5
        assert s.name == "fouls"

    def test_fouls_custom_line_225(self):
        s = FoulsStrategy(line=22.5)
        assert s.line == 22.5
        assert s.name == "fouls_over_225"

    def test_fouls_custom_line_265(self):
        s = FoulsStrategy(line=26.5)
        assert s.line == 26.5
        assert s.name == "fouls_over_265"


class TestTargetCreation:
    """Test that create_target uses the correct line."""

    @pytest.fixture
    def cards_df(self):
        return pd.DataFrame({
            'total_cards': [3, 4, 5, 6, 7],
        })

    @pytest.fixture
    def corners_df(self):
        return pd.DataFrame({
            'total_corners': [7, 8, 9, 10, 11, 12],
        })

    def test_cards_default_target(self, cards_df):
        """Cards default (4.5): 5,6,7 > 4.5 -> 3 positives."""
        s = CardsStrategy()
        df, col = s.create_target(cards_df)
        assert col == 'target'
        assert df['target'].sum() == 3  # 5, 6, 7

    def test_cards_over_35_target(self, cards_df):
        """Cards 3.5: 4,5,6,7 > 3.5 -> 4 positives."""
        s = CardsStrategy(line=3.5)
        df, col = s.create_target(cards_df)
        assert df['target'].sum() == 4

    def test_cards_over_55_target(self, cards_df):
        """Cards 5.5: 6,7 > 5.5 -> 2 positives."""
        s = CardsStrategy(line=5.5)
        df, col = s.create_target(cards_df)
        assert df['target'].sum() == 2

    def test_cards_over_65_target(self, cards_df):
        """Cards 6.5: 7 > 6.5 -> 1 positive."""
        s = CardsStrategy(line=6.5)
        df, col = s.create_target(cards_df)
        assert df['target'].sum() == 1

    def test_corners_default_target(self, corners_df):
        """Corners default (9.5): 10,11,12 > 9.5 -> 3 positives."""
        s = CornersStrategy()
        df, col = s.create_target(corners_df)
        assert df['target'].sum() == 3

    def test_corners_over_85_target(self, corners_df):
        """Corners 8.5: 9,10,11,12 > 8.5 -> 4 positives."""
        s = CornersStrategy(line=8.5)
        df, col = s.create_target(corners_df)
        assert df['target'].sum() == 4

    def test_corners_over_105_target(self, corners_df):
        """Corners 10.5: 11,12 > 10.5 -> 2 positives."""
        s = CornersStrategy(line=10.5)
        df, col = s.create_target(corners_df)
        assert df['target'].sum() == 2


class TestStrategyRegistry:
    """Test that the registry and factory correctly dispatch line variants."""

    def test_get_strategy_cards_default(self):
        s = get_strategy("cards")
        assert isinstance(s, CardsStrategy)
        assert s.line == 4.5
        assert s.name == "cards"

    def test_get_strategy_cards_over_35(self):
        s = get_strategy("cards_over_35")
        assert isinstance(s, CardsStrategy)
        assert s.line == 3.5
        assert s.name == "cards_over_35"

    def test_get_strategy_cards_over_55(self):
        s = get_strategy("cards_over_55")
        assert isinstance(s, CardsStrategy)
        assert s.line == 5.5

    def test_get_strategy_cards_over_65(self):
        s = get_strategy("cards_over_65")
        assert isinstance(s, CardsStrategy)
        assert s.line == 6.5

    def test_get_strategy_corners_over_85(self):
        s = get_strategy("corners_over_85")
        assert isinstance(s, CornersStrategy)
        assert s.line == 8.5

    def test_get_strategy_corners_over_105(self):
        s = get_strategy("corners_over_105")
        assert isinstance(s, CornersStrategy)
        assert s.line == 10.5

    def test_get_strategy_corners_over_115(self):
        s = get_strategy("corners_over_115")
        assert isinstance(s, CornersStrategy)
        assert s.line == 11.5

    def test_get_strategy_shots_over_225(self):
        s = get_strategy("shots_over_225")
        assert isinstance(s, ShotsStrategy)
        assert s.line == 22.5

    def test_get_strategy_fouls_over_265(self):
        s = get_strategy("fouls_over_265")
        assert isinstance(s, FoulsStrategy)
        assert s.line == 26.5

    def test_get_strategy_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("cards_over_99")

    def test_all_line_variants_in_registry(self):
        """Every entry in NICHE_LINE_LOOKUP has a registry entry."""
        for name in NICHE_LINE_LOOKUP:
            assert name in STRATEGY_REGISTRY, f"{name} missing from STRATEGY_REGISTRY"


class TestDefaultLineConsistency:
    """Verify niche strategy default_line matches BET_TYPES target_line."""

    def test_cards_default_line_matches_sniper(self):
        from experiments.run_sniper_optimization import BET_TYPES
        assert CardsStrategy().default_line == BET_TYPES["cards"]["target_line"]

    def test_corners_default_line_matches_sniper(self):
        from experiments.run_sniper_optimization import BET_TYPES
        assert CornersStrategy().default_line == BET_TYPES["corners"]["target_line"]

    def test_shots_default_line_matches_sniper(self):
        from experiments.run_sniper_optimization import BET_TYPES
        assert ShotsStrategy().default_line == BET_TYPES["shots"]["target_line"]

    def test_fouls_default_line_matches_sniper(self):
        from experiments.run_sniper_optimization import BET_TYPES
        assert FoulsStrategy().default_line == BET_TYPES["fouls"]["target_line"]


class TestOddsColumn:
    """Test that each variant returns the correct default_odds_column."""

    def test_cards_default_odds_column(self):
        assert CardsStrategy().default_odds_column == "cards_over_4_5"

    def test_cards_over_35_odds_column(self):
        assert CardsStrategy(line=3.5).default_odds_column == "cards_over_3_5"

    def test_corners_default_odds_column(self):
        assert CornersStrategy().default_odds_column == "corners_over_9_5"

    def test_corners_over_105_odds_column(self):
        assert CornersStrategy(line=10.5).default_odds_column == "corners_over_10_5"

    def test_shots_default_odds_column(self):
        assert ShotsStrategy().default_odds_column == "shots_over_24_5"

    def test_fouls_default_odds_column(self):
        assert FoulsStrategy().default_odds_column == "fouls_over_24_5"


class TestBaseMarketMap:
    """Test BASE_MARKET_MAP completeness."""

    def test_all_line_variants_have_base_market(self):
        """Every line variant in NICHE_LINE_LOOKUP has a BASE_MARKET_MAP entry."""
        for name in NICHE_LINE_LOOKUP:
            assert name in BASE_MARKET_MAP, f"{name} missing from BASE_MARKET_MAP"

    def test_base_markets_are_valid(self):
        """Every base market in BASE_MARKET_MAP is a valid registry entry."""
        for variant, base in BASE_MARKET_MAP.items():
            assert base in STRATEGY_REGISTRY, f"Base market '{base}' for '{variant}' not in registry"

    def test_sniper_base_market_map_matches(self):
        """Sniper optimization BASE_MARKET_MAP matches betting_strategies one."""
        from experiments.run_sniper_optimization import BASE_MARKET_MAP as SNIPER_MAP
        assert SNIPER_MAP == BASE_MARKET_MAP

    def test_base_market_map_values(self):
        """Spot-check some mappings."""
        assert BASE_MARKET_MAP['cards_over_35'] == 'cards'
        assert BASE_MARKET_MAP['corners_over_105'] == 'corners'
        assert BASE_MARKET_MAP['shots_over_225'] == 'shots'
        assert BASE_MARKET_MAP['fouls_over_265'] == 'fouls'


class TestBetSide:
    """Test bet_side property includes line info."""

    def test_cards_default_bet_side(self):
        assert CardsStrategy().bet_side == "cards over 4.5"

    def test_cards_over_35_bet_side(self):
        assert CardsStrategy(line=3.5).bet_side == "cards over 3.5"

    def test_corners_default_bet_side(self):
        assert CornersStrategy().bet_side == "corners over 9.5"

    def test_fouls_default_bet_side(self):
        assert FoulsStrategy().bet_side == "fouls over 24.5"
