"""Unit tests for LineupConfidenceAdjuster."""

import numpy as np
import pandas as pd
import pytest

from src.ml.confidence_adjuster import (
    CATEGORY_WEIGHTS,
    ConfidenceAdjustment,
    LineupAnalysis,
    LineupConfidenceAdjuster,
    adjust_single_prediction,
)


def _full_lineup(names):
    """Build lineup dict from a list of player names."""
    return {
        'formation': '4-3-3',
        'startXI': [{'player': {'name': n}} for n in names],
    }


# Liverpool full-strength lineup (team_id=40)
LIVERPOOL_LINEUP = _full_lineup([
    'Alisson', 'Trent Alexander-Arnold', 'Virgil van Dijk',
    'Ibrahima Konate', 'Andrew Robertson', 'Dominik Szoboszlai',
    'Alexis Mac Allister', 'Curtis Jones', 'Mohamed Salah',
    'Darwin Nunez', 'Luis Diaz',
])

# Man City missing Haaland and KDB (team_id=50)
CITY_WEAKENED = _full_lineup([
    'Ederson', 'Kyle Walker', 'Ruben Dias', 'John Stones',
    'Josko Gvardiol', 'Rodri', 'Mateo Kovacic', 'Bernardo Silva',
    'Jack Grealish', 'Phil Foden', 'Julian Alvarez',
])


class TestLineupAnalysis:
    def test_full_strength_score_1(self):
        adjuster = LineupConfidenceAdjuster()
        analysis = adjuster.analyze_lineup(40, 'Liverpool', LIVERPOOL_LINEUP)
        assert analysis.strength_score == 1.0
        assert analysis.key_players_missing == []

    def test_missing_key_players_detected(self):
        adjuster = LineupConfidenceAdjuster()
        analysis = adjuster.analyze_lineup(50, 'Man City', CITY_WEAKENED)
        assert 'Erling Haaland' in analysis.key_players_missing
        assert 'Kevin De Bruyne' in analysis.key_players_missing
        assert analysis.strength_score < 1.0

    def test_unknown_team_returns_full_strength(self):
        adjuster = LineupConfidenceAdjuster()
        analysis = adjuster.analyze_lineup(99999, 'Unknown FC', _full_lineup(['Player A']))
        assert analysis.strength_score == 1.0
        assert analysis.key_players_available == []
        assert analysis.key_players_missing == []

    def test_formation_extracted(self):
        adjuster = LineupConfidenceAdjuster()
        analysis = adjuster.analyze_lineup(40, 'Liverpool', LIVERPOOL_LINEUP)
        assert analysis.formation == '4-3-3'

    def test_missing_impact_by_category(self):
        adjuster = LineupConfidenceAdjuster()
        analysis = adjuster.analyze_lineup(50, 'Man City', CITY_WEAKENED)
        # Haaland is 'scoring', KDB is 'core' — both categories should have impact
        assert 'scoring' in analysis.missing_impact or 'core' in analysis.missing_impact

    def test_custom_key_players(self):
        custom = {1: {'core': ['Player X'], 'scoring': ['Player Y']}}
        adjuster = LineupConfidenceAdjuster(key_players=custom)
        lineup = _full_lineup(['Player X', 'Some Other'])
        analysis = adjuster.analyze_lineup(1, 'Team A', lineup)
        assert 'Player X' in analysis.key_players_available
        assert 'Player Y' in analysis.key_players_missing

    def test_empty_lineup(self):
        adjuster = LineupConfidenceAdjuster()
        analysis = adjuster.analyze_lineup(40, 'Liverpool', {})
        # All key players missing when lineup is empty
        assert len(analysis.key_players_missing) > 0
        assert analysis.strength_score < 1.0


class TestCalculateAdjustment:
    def _make_analyses(self, home_strength=1.0, away_strength=1.0):
        home = LineupAnalysis(
            team_id=40, team_name='Home',
            key_players_available=[], key_players_missing=[],
            formation='4-3-3', strength_score=home_strength,
            missing_impact={},
        )
        away = LineupAnalysis(
            team_id=50, team_name='Away',
            key_players_available=[], key_players_missing=[],
            formation='4-3-3', strength_score=away_strength,
            missing_impact={},
        )
        return home, away

    def test_no_missing_players_neutral(self):
        adjuster = LineupConfidenceAdjuster()
        home, away = self._make_analyses()
        adj = adjuster.calculate_adjustment(1, 'home_win', 0.50, home, away)
        assert adj.confidence_change == 'neutral'
        assert adj.adjusted_probability == pytest.approx(0.50, abs=0.01)

    def test_adjustment_capped_at_max(self):
        adjuster = LineupConfidenceAdjuster(max_adjustment=0.10)
        # Create analyses with extreme missing impact
        home = LineupAnalysis(
            team_id=40, team_name='Home',
            key_players_available=[], key_players_missing=['A', 'B', 'C'],
            formation='4-3-3', strength_score=0.3,
            missing_impact={'core': 0.9, 'scoring': 0.9, 'defensive': 0.9},
        )
        away = LineupAnalysis(
            team_id=50, team_name='Away',
            key_players_available=[], key_players_missing=[],
            formation='4-3-3', strength_score=1.0,
            missing_impact={},
        )
        adj = adjuster.calculate_adjustment(1, 'home_win', 0.50, home, away)
        # Factor should be clamped to 1 +/- max_adjustment
        assert adj.adjustment_factor >= 1.0 - 0.10
        assert adj.adjustment_factor <= 1.0 + 0.10

    def test_probability_clipped_to_valid_range(self):
        adjuster = LineupConfidenceAdjuster(max_adjustment=0.50)
        home, away = self._make_analyses()
        # Very high original prob
        adj = adjuster.calculate_adjustment(1, 'home_win', 0.99, home, away)
        assert 0.01 <= adj.adjusted_probability <= 0.99

    def test_home_win_reduces_when_home_weakened(self):
        adjuster = LineupConfidenceAdjuster(key_player_threshold=1)
        home = LineupAnalysis(
            team_id=40, team_name='Home',
            key_players_available=[],
            key_players_missing=['Salah', 'Van Dijk'],
            formation='4-3-3', strength_score=0.5,
            missing_impact={'scoring': 0.5, 'core': 0.5},
        )
        away = LineupAnalysis(
            team_id=50, team_name='Away',
            key_players_available=['Haaland'],
            key_players_missing=[],
            formation='4-3-3', strength_score=1.0,
            missing_impact={},
        )
        adj = adjuster.calculate_adjustment(1, 'home_win', 0.60, home, away)
        assert adj.adjusted_probability < 0.60
        assert adj.confidence_change == 'reduce'

    def test_btts_reduces_when_scorers_missing(self):
        adjuster = LineupConfidenceAdjuster()
        home = LineupAnalysis(
            team_id=40, team_name='Home',
            key_players_available=[], key_players_missing=['Salah'],
            formation='4-3-3', strength_score=0.8,
            missing_impact={'scoring': 0.5},
        )
        away = LineupAnalysis(
            team_id=50, team_name='Away',
            key_players_available=[], key_players_missing=[],
            formation='4-3-3', strength_score=1.0,
            missing_impact={},
        )
        adj = adjuster.calculate_adjustment(1, 'btts', 0.60, home, away)
        assert adj.adjusted_probability < 0.60

    def test_over25_increases_when_defenders_missing(self):
        adjuster = LineupConfidenceAdjuster()
        home = LineupAnalysis(
            team_id=40, team_name='Home',
            key_players_available=[], key_players_missing=[],
            formation='4-3-3', strength_score=1.0,
            missing_impact={},
        )
        away = LineupAnalysis(
            team_id=50, team_name='Away',
            key_players_available=[], key_players_missing=['Dias'],
            formation='4-3-3', strength_score=0.8,
            missing_impact={'defensive': 0.5},
        )
        adj = adjuster.calculate_adjustment(1, 'over25', 0.50, home, away)
        assert adj.adjusted_probability > 0.50

    def test_unknown_market_no_change(self):
        adjuster = LineupConfidenceAdjuster()
        home, away = self._make_analyses()
        adj = adjuster.calculate_adjustment(1, 'exotic_market', 0.50, home, away)
        assert adj.adjusted_probability == pytest.approx(0.50, abs=0.01)

    def test_returns_correct_dataclass(self):
        adjuster = LineupConfidenceAdjuster()
        home, away = self._make_analyses()
        adj = adjuster.calculate_adjustment(1, 'home_win', 0.50, home, away)
        assert isinstance(adj, ConfidenceAdjustment)
        assert adj.fixture_id == 1
        assert adj.market == 'home_win'


class TestAdjustPredictions:
    def test_adjust_predictions_adds_columns(self):
        adjuster = LineupConfidenceAdjuster()
        preds = pd.DataFrame({
            'fixture_id': [1],
            'home_team_id': [40],
            'away_team_id': [50],
            'home_team_name': ['Liverpool'],
            'away_team_name': ['Man City'],
            'home_win_prob': [0.55],
        })
        lineups = {
            1: {
                'home': LIVERPOOL_LINEUP,
                'away': CITY_WEAKENED,
            }
        }
        result = adjuster.adjust_predictions(preds, lineups, markets=['home_win'])
        assert 'home_win_prob_adj' in result.columns
        assert 'home_strength' in result.columns
        assert len(result) == 1

    def test_adjust_predictions_no_lineups(self):
        adjuster = LineupConfidenceAdjuster()
        preds = pd.DataFrame({
            'fixture_id': [1],
            'home_team_id': [40],
            'away_team_id': [50],
            'home_win_prob': [0.55],
        })
        result = adjuster.adjust_predictions(preds, {}, markets=['home_win'])
        assert len(result) == 1


class TestAdjustSinglePrediction:
    def test_convenience_function_returns_adjustment(self):
        adj = adjust_single_prediction(
            fixture_id=1,
            market='home_win',
            original_prob=0.55,
            home_team_id=40,
            away_team_id=50,
            home_lineup=LIVERPOOL_LINEUP,
            away_lineup=CITY_WEAKENED,
        )
        assert isinstance(adj, ConfidenceAdjustment)
        assert adj.original_probability == 0.55
