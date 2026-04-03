"""Tests that known dead columns are not produced by feature engineers."""
import pytest


DEAD_COLUMNS = [
    'implied_total_goals',
    'implied_goal_supremacy',
    'abs_goal_supremacy',
]


class TestDeadColumns:
    def test_cross_market_does_not_produce_dead_columns(self):
        """CrossMarketFeatureEngineer must not create 100%-NaN columns."""
        from src.features.engineers.cross_market import CrossMarketFeatureEngineer
        import pandas as pd

        eng = CrossMarketFeatureEngineer()
        matches = pd.DataFrame({
            'fixture_id': [1, 2],
            'date': ['2025-01-01', '2025-01-02'],
            'home_team': ['A', 'C'],
            'away_team': ['B', 'D'],
            'home_goals_scored_ema': [1.5, 1.8],
            'away_goals_scored_ema': [1.2, 1.0],
        })
        result = eng.create_features({'matches': matches})
        for col in DEAD_COLUMNS:
            assert col not in result.columns, f"Dead column {col} still produced"
