"""Pytest configuration and shared fixtures."""
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_fixture_data():
    """Sample fixture data for testing."""
    return {
        'fixture': {
            'id': 12345,
            'date': '2024-01-01T15:00:00+00:00',
            'timestamp': 1704114000,
            'referee': 'John Doe',
            'venue': {'id': 1, 'name': 'Stadium A'},
            'status': {'short': 'FT'}
        },
        'league': {
            'id': 39,
            'round': 'Regular Season - 1'
        },
        'teams': {
            'home': {'id': 100, 'name': 'Team A'},
            'away': {'id': 200, 'name': 'Team B'}
        },
        'score': {
            'fulltime': {'home': 2, 'away': 1},
            'halftime': {'home': 1, 'away': 0}
        }
    }


@pytest.fixture
def sample_matches_df():
    """Sample matches DataFrame for testing."""
    import pandas as pd

    return pd.DataFrame({
        'fixture_id': [1, 2, 3, 4, 5],
        'date': pd.to_datetime([
            '2024-01-01', '2024-01-08', '2024-01-15',
            '2024-01-22', '2024-01-29'
        ]),
        'home_team_id': [100, 200, 100, 300, 100],
        'away_team_id': [200, 300, 300, 100, 200],
        'home_team_name': ['Team A', 'Team B', 'Team A', 'Team C', 'Team A'],
        'away_team_name': ['Team B', 'Team C', 'Team C', 'Team A', 'Team B'],
        'ft_home': [2, 1, 3, 0, 1],
        'ft_away': [1, 1, 0, 2, 1],
        'round': ['Round 1', 'Round 1', 'Round 2', 'Round 2', 'Round 3']
    })


@pytest.fixture
def sample_player_stats_df():
    """Sample player stats DataFrame for testing."""
    import pandas as pd

    return pd.DataFrame({
        'fixture_id': [1, 1, 1, 1],
        'player_id': [1001, 1002, 2001, 2002],
        'player_name': ['Player A1', 'Player A2', 'Player B1', 'Player B2'],
        'team_id': [100, 100, 200, 200],
        'team_name': ['Team A', 'Team A', 'Team B', 'Team B'],
        'minutes': [90, 75, 90, 60],
        'goals': [1, 0, 1, 0],
        'assists': [0, 1, 0, 1],
        'rating': [7.5, 6.8, 7.2, 6.5],
        'shots_total': [5, 2, 3, 1],
        'shots_on': [3, 1, 2, 0],
        'passes_total': [45, 38, 42, 35],
        'passes_key': [3, 2, 1, 1],
        'passes_accuracy': [85, 80, 82, 78],
        'tackles_total': [2, 3, 4, 2],
        'fouls_committed': [1, 2, 1, 0],
        'yellow_cards': [0, 1, 0, 0],
        'red_cards': [0, 0, 0, 0]
    })


@pytest.fixture
def config_local_path():
    """Path to local config file."""
    return project_root / "config" / "local.yaml"


@pytest.fixture
def config_prod_path():
    """Path to prod config file."""
    return project_root / "config" / "prod.yaml"
