"""Feature engineering - Head-to-head and derby features."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer


class HeadToHeadFeatureEngineer(BaseFeatureEngineer):
    """Creates features related to history of face-to-face matches."""

    def __init__(self, n_h2h: int = 3):
        """
        Args:
            n_h2h: how many matches take
        """
        self.n_h2h = n_h2h

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Creates features head-to-head.

        Args:
            data: dict {name:DataFrame}

        Returns:
            new DataFrame with H2H features
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date')

        features_list = []

        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']
            match_date = match['date']

            h2h_matches = matches[
                (matches['date'] < match_date) &
                (
                        ((matches['home_team_id'] == home_id) & (matches['away_team_id'] == away_id)) |
                        ((matches['home_team_id'] == away_id) & (matches['away_team_id'] == home_id))
                )
                ].tail(self.n_h2h)

            if len(h2h_matches) == 0:
                h2h_home_wins = h2h_draws = h2h_away_wins = 0
                h2h_avg_goals = 0
            else:
                h2h_home_wins = h2h_draws = h2h_away_wins = 0
                total_goals = 0

                for _, h2h in h2h_matches.iterrows():
                    if h2h['home_team_id'] == home_id:
                        if h2h['ft_home'] > h2h['ft_away']:
                            h2h_home_wins += 1
                        elif h2h['ft_home'] == h2h['ft_away']:
                            h2h_draws += 1
                        else:
                            h2h_away_wins += 1
                    else:
                        if h2h['ft_away'] > h2h['ft_home']:
                            h2h_home_wins += 1
                        elif h2h['ft_away'] == h2h['ft_home']:
                            h2h_draws += 1
                        else:
                            h2h_away_wins += 1

                    total_goals += h2h['ft_home'] + h2h['ft_away']

                h2h_avg_goals = total_goals / len(h2h_matches)

            features = {
                'fixture_id': match['fixture_id'],
                'h2h_home_wins': h2h_home_wins,
                'h2h_draws': h2h_draws,
                'h2h_away_wins': h2h_away_wins,
                'h2h_avg_goals': h2h_avg_goals
            }

            features_list.append(features)

        print(f"Created {len(features_list)} head-to-head features (last {self.n_h2h} matches)")
        return pd.DataFrame(features_list)



class DerbyFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features for local derby matches.

    Derbies are high-stakes matches where:
    - Form often goes out the window
    - Home advantage is amplified
    - Results are less predictable
    """

    # Premier League derbies
    DERBY_PAIRS = {
        # Manchester Derby
        ('Manchester United', 'Manchester City'),
        ('Manchester City', 'Manchester United'),
        # North London Derby
        ('Arsenal', 'Tottenham'),
        ('Tottenham', 'Arsenal'),
        # Merseyside Derby
        ('Liverpool', 'Everton'),
        ('Everton', 'Liverpool'),
        # North West Derby
        ('Liverpool', 'Manchester United'),
        ('Manchester United', 'Liverpool'),
        # London Derbies
        ('Chelsea', 'Arsenal'),
        ('Arsenal', 'Chelsea'),
        ('Chelsea', 'Tottenham'),
        ('Tottenham', 'Chelsea'),
        ('West Ham', 'Tottenham'),
        ('Tottenham', 'West Ham'),
        ('Chelsea', 'West Ham'),
        ('West Ham', 'Chelsea'),
        # Yorkshire
        ('Leeds', 'Sheffield'),
        ('Sheffield', 'Leeds'),
        # Midlands
        ('Aston Villa', 'Birmingham'),
        ('Birmingham', 'Aston Villa'),
        ('Aston Villa', 'Wolves'),
        ('Wolves', 'Aston Villa'),
        # Others
        ('Newcastle', 'Sunderland'),
        ('Sunderland', 'Newcastle'),
        ('Brighton', 'Crystal Palace'),
        ('Crystal Palace', 'Brighton'),
    }

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate derby features.
        """
        matches = data['matches'].copy()
        features_list = []

        for idx, match in matches.iterrows():
            home_name = match['home_team_name']
            away_name = match['away_team_name']

            # Check if derby
            is_derby = (home_name, away_name) in self.DERBY_PAIRS

            # Check for same-city teams (simplified)
            same_city = self._same_city(home_name, away_name)

            features = {
                'fixture_id': match['fixture_id'],
                'is_derby': 1 if is_derby else 0,
                'is_same_city': 1 if same_city else 0,
                'is_rivalry': 1 if (is_derby or same_city) else 0,
            }
            features_list.append(features)

        derby_count = sum(1 for f in features_list if f['is_derby'] == 1)
        print(f"Created {len(features_list)} derby features ({derby_count} derbies found)")
        return pd.DataFrame(features_list)

    def _same_city(self, home: str, away: str) -> bool:
        """Check if teams are from the same city."""
        city_teams = {
            'London': ['Arsenal', 'Chelsea', 'Tottenham', 'West Ham', 'Crystal Palace',
                       'Fulham', 'Brentford', 'QPR', 'Charlton', 'Millwall'],
            'Manchester': ['Manchester United', 'Manchester City'],
            'Liverpool': ['Liverpool', 'Everton'],
            'Birmingham': ['Aston Villa', 'Birmingham', 'West Brom', 'Wolves'],
            'Sheffield': ['Sheffield United', 'Sheffield Wednesday'],
        }

        for city, teams in city_teams.items():
            if home in teams and away in teams:
                return True
        return False


