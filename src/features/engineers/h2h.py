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

        # La Liga
        ('Real Madrid', 'Barcelona'),
        ('Barcelona', 'Real Madrid'),
        ('Real Madrid', 'Atletico Madrid'),
        ('Atletico Madrid', 'Real Madrid'),
        ('Barcelona', 'Atletico Madrid'),
        ('Atletico Madrid', 'Barcelona'),
        ('Real Betis', 'Sevilla'),
        ('Sevilla', 'Real Betis'),

        # Serie A
        ('AC Milan', 'Inter'),
        ('Inter', 'AC Milan'),
        ('AS Roma', 'Lazio'),
        ('Lazio', 'AS Roma'),
        ('Juventus', 'Torino'),
        ('Torino', 'Juventus'),
        ('Juventus', 'Inter'),
        ('Inter', 'Juventus'),

        # Bundesliga
        ('Borussia Dortmund', 'Schalke 04'),
        ('Schalke 04', 'Borussia Dortmund'),
        ('Bayern Munich', 'Borussia Dortmund'),
        ('Borussia Dortmund', 'Bayern Munich'),
        ('Hamburger SV', 'Werder Bremen'),
        ('Werder Bremen', 'Hamburger SV'),

        # Turkish Super Lig
        ('Galatasaray', 'Fenerbahçe'),
        ('Fenerbahçe', 'Galatasaray'),
        ('Galatasaray', 'Beşiktaş'),
        ('Beşiktaş', 'Galatasaray'),
        ('Fenerbahçe', 'Beşiktaş'),
        ('Beşiktaş', 'Fenerbahçe'),
        ('Galatasaray', 'Trabzonspor'),
        ('Trabzonspor', 'Galatasaray'),
        ('Fenerbahçe', 'Trabzonspor'),
        ('Trabzonspor', 'Fenerbahçe'),

        # Portuguese Liga
        ('Benfica', 'Sporting CP'),
        ('Sporting CP', 'Benfica'),
        ('Benfica', 'FC Porto'),
        ('FC Porto', 'Benfica'),
        ('Sporting CP', 'FC Porto'),
        ('FC Porto', 'Sporting CP'),
        ('Boavista', 'FC Porto'),
        ('FC Porto', 'Boavista'),
        ('Guimaraes', 'SC Braga'),
        ('SC Braga', 'Guimaraes'),

        # Scottish Premiership
        ('Celtic', 'Rangers'),
        ('Rangers', 'Celtic'),
        ('Heart Of Midlothian', 'Hibernian'),
        ('Hibernian', 'Heart Of Midlothian'),
        ('Aberdeen', 'Rangers'),
        ('Rangers', 'Aberdeen'),
        ('Aberdeen', 'Celtic'),
        ('Celtic', 'Aberdeen'),
        ('Dundee', 'Dundee Utd'),
        ('Dundee Utd', 'Dundee'),

        # Belgian Pro League
        ('Anderlecht', 'Club Brugge KV'),
        ('Club Brugge KV', 'Anderlecht'),
        ('Anderlecht', 'Standard Liege'),
        ('Standard Liege', 'Anderlecht'),
        ('Club Brugge KV', 'Cercle Brugge'),
        ('Cercle Brugge', 'Club Brugge KV'),
        ('Gent', 'Club Brugge KV'),
        ('Club Brugge KV', 'Gent'),
        ('Antwerp', 'Beerschot VA'),
        ('Beerschot VA', 'Antwerp'),

        # Eredivisie
        ('Ajax', 'Feyenoord'),
        ('Feyenoord', 'Ajax'),
        ('Ajax', 'PSV Eindhoven'),
        ('PSV Eindhoven', 'Ajax'),
        ('Feyenoord', 'PSV Eindhoven'),
        ('PSV Eindhoven', 'Feyenoord'),
        ('Ajax', 'Utrecht'),
        ('Utrecht', 'Ajax'),
        ('Feyenoord', 'Sparta Rotterdam'),
        ('Sparta Rotterdam', 'Feyenoord'),
        ('NEC Nijmegen', 'Vitesse'),
        ('Vitesse', 'NEC Nijmegen'),
        ('Twente', 'Heracles'),
        ('Heracles', 'Twente'),
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
            # Premier League
            'London': ['Arsenal', 'Chelsea', 'Tottenham', 'West Ham', 'Crystal Palace',
                       'Fulham', 'Brentford', 'QPR', 'Charlton', 'Millwall'],
            'Manchester': ['Manchester United', 'Manchester City'],
            'Liverpool': ['Liverpool', 'Everton'],
            'Birmingham': ['Aston Villa', 'Birmingham', 'West Brom', 'Wolves'],
            'Sheffield': ['Sheffield United', 'Sheffield Wednesday'],
            # La Liga
            'Madrid': ['Real Madrid', 'Atletico Madrid', 'Getafe', 'Rayo Vallecano', 'Leganes'],
            'Seville': ['Sevilla', 'Real Betis'],
            'Barcelona_city': ['Barcelona', 'Espanyol'],
            # Serie A
            'Milan': ['AC Milan', 'Inter'],
            'Rome': ['AS Roma', 'Lazio'],
            'Turin': ['Juventus', 'Torino'],
            'Genoa_city': ['Genoa', 'Sampdoria'],
            # Bundesliga
            'Munich_city': ['Bayern Munich', '1860 Munich'],
            # Turkish Super Lig
            'Istanbul': ['Galatasaray', 'Fenerbahçe', 'Beşiktaş', 'Başakşehir',
                         'İstanbulspor', 'Fatih Karagümrük', 'Kasımpaşa'],
            'Ankara': ['Ankaragücü', 'Gençlerbirliği'],
            # Portuguese Liga
            'Lisbon': ['Benfica', 'Sporting CP', 'Belenenses'],
            'Porto_city': ['FC Porto', 'Boavista'],
            # Scottish Premiership
            'Glasgow': ['Celtic', 'Rangers'],
            'Edinburgh': ['Heart Of Midlothian', 'Hibernian'],
            'Dundee_city': ['Dundee', 'Dundee Utd'],
            # Belgian Pro League
            'Bruges': ['Club Brugge KV', 'Cercle Brugge'],
            'Brussels': ['Anderlecht', 'Union St. Gilloise'],
            'Antwerp_city': ['Antwerp', 'Beerschot VA'],
            # Eredivisie
            'Rotterdam': ['Feyenoord', 'Sparta Rotterdam', 'Excelsior'],
            'Eindhoven_city': ['PSV Eindhoven', 'FC Eindhoven'],
        }

        for city, teams in city_teams.items():
            if home in teams and away in teams:
                return True
        return False


